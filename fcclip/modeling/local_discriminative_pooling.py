"""Local discriminative pooling for FC-CLIP region representations."""

from typing import Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


def combine_ldp_non_void_logits(
    ldp_logits: torch.Tensor,
    base_logits: torch.Tensor,
) -> torch.Tensor:
    """Use LDP class logits while preserving FC-CLIP's learned void logit."""
    if ldp_logits.ndim != 3:
        raise ValueError(
            "ldp_logits must have shape [B, Q, K+1], "
            f"got {tuple(ldp_logits.shape)}"
        )
    if base_logits.shape != ldp_logits.shape:
        raise ValueError(
            "base_logits and ldp_logits must have the same shape, "
            f"got {tuple(base_logits.shape)} and {tuple(ldp_logits.shape)}"
        )
    if ldp_logits.shape[-1] < 2:
        raise ValueError(
            "classification logits must contain at least one class and one void "
            f"entry, got {ldp_logits.shape[-1]} entries"
        )
    return torch.cat((ldp_logits[..., :-1], base_logits[..., -1:]), dim=-1)


def validate_ldp_inputs(
    structure_features: torch.Tensor,
    clip_features: torch.Tensor,
    *,
    structure_dim: int,
    embed_dim: int,
) -> None:
    """Validate the shared two-input LDP contract."""
    if structure_features.ndim != 3:
        raise ValueError(
            "structure_features must have shape [B, Q, C], "
            f"got {tuple(structure_features.shape)}"
        )
    if clip_features.ndim != 3:
        raise ValueError(
            "clip_features must have shape [B, Q, C], "
            f"got {tuple(clip_features.shape)}"
        )
    if structure_features.shape[:2] != clip_features.shape[:2]:
        raise ValueError(
            "structure and CLIP features must have matching batch/query dimensions, "
            f"got {tuple(structure_features.shape[:2])} and "
            f"{tuple(clip_features.shape[:2])}"
        )
    if structure_features.shape[-1] != structure_dim:
        raise ValueError(
            "structure feature dimension does not match the configured dimension, "
            f"got {structure_features.shape[-1]} and {structure_dim}"
        )
    if clip_features.shape[-1] != embed_dim:
        raise ValueError(
            "CLIP feature dimension does not match the configured embedding dimension, "
            f"got {clip_features.shape[-1]} and {embed_dim}"
        )
    if structure_features.device != clip_features.device:
        raise ValueError(
            "structure and CLIP features must be on the same device, "
            f"got {structure_features.device} and {clip_features.device}"
        )
    if structure_features.dtype != clip_features.dtype:
        raise ValueError(
            "structure and CLIP features must have the same dtype, "
            f"got {structure_features.dtype} and {clip_features.dtype}"
        )


def validate_ldp_configuration(
    *,
    structure_dim: int,
    embed_dim: int,
    num_heads: int,
    dropout: float,
) -> None:
    """Validate dimensions shared by all LDP topologies."""
    if structure_dim <= 0:
        raise ValueError(f"structure_dim must be positive, got {structure_dim}")
    if embed_dim <= 0:
        raise ValueError(f"embed_dim must be positive, got {embed_dim}")
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if embed_dim % num_heads != 0:
        raise ValueError(
            "embed_dim must be divisible by num_heads, "
            f"got embed_dim={embed_dim}, num_heads={num_heads}"
        )
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}")


class LocalDiscriminativePooling(nn.Module):
    """Fuse all decoder structure tokens with frozen CLIP region tokens."""

    def __init__(
        self,
        *,
        structure_dim: int,
        embed_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        validate_ldp_configuration(
            structure_dim=structure_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.structure_dim = structure_dim
        self.embed_dim = embed_dim
        self.structure_projection = nn.Linear(structure_dim, embed_dim)
        self.clip_projection = nn.Linear(embed_dim, embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=2 * embed_dim,
            vdim=embed_dim,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(embed_dim)

        nn.init.xavier_uniform_(self.structure_projection.weight)
        nn.init.zeros_(self.structure_projection.bias)
        nn.init.eye_(self.clip_projection.weight)
        nn.init.zeros_(self.clip_projection.bias)

    def forward(
        self,
        structure_features: torch.Tensor,
        clip_features: torch.Tensor,
    ) -> torch.Tensor:
        validate_ldp_inputs(
            structure_features,
            clip_features,
            structure_dim=self.structure_dim,
            embed_dim=self.embed_dim,
        )
        structure_tokens = self.structure_projection(structure_features)
        clip_tokens = self.clip_projection(clip_features)
        joint_keys = torch.cat((structure_tokens, clip_tokens), dim=-1)
        attention_delta, _ = self.attention(
            query=clip_tokens,
            key=joint_keys,
            value=structure_tokens,
            need_weights=False,
        )
        return self.output_norm(clip_tokens + attention_delta)


class PairedGatedCorrection(nn.Module):
    """Apply a per-mask, per-head structure correction to a CLIP anchor.

    The two keys are the paired CLIP and structure tokens. Only the structure
    token has a value, so attending to the CLIP key means keeping the semantic
    anchor unchanged before the final LayerNorm.
    """

    def __init__(
        self,
        *,
        structure_dim: int,
        embed_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        validate_ldp_configuration(
            structure_dim=structure_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.structure_dim = structure_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.structure_projection = nn.Linear(structure_dim, embed_dim)
        self.clip_projection = nn.Linear(embed_dim, embed_dim)
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.structure_value_projection = nn.Linear(embed_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)

        nn.init.xavier_uniform_(self.structure_projection.weight)
        nn.init.zeros_(self.structure_projection.bias)
        nn.init.eye_(self.clip_projection.weight)
        nn.init.zeros_(self.clip_projection.bias)
        for projection in (
            self.query_projection,
            self.key_projection,
            self.structure_value_projection,
        ):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
        nn.init.zeros_(self.output_projection.bias)

    def _project_attention_inputs(
        self,
        structure_features: torch.Tensor,
        clip_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        validate_ldp_inputs(
            structure_features,
            clip_features,
            structure_dim=self.structure_dim,
            embed_dim=self.embed_dim,
        )
        batch_size, num_queries, _ = clip_features.shape
        structure_tokens = self.structure_projection(structure_features)
        clip_tokens = self.clip_projection(clip_features)
        queries = self.query_projection(clip_tokens).reshape(
            batch_size, num_queries, self.num_heads, self.head_dim
        )
        paired_tokens = torch.stack((clip_tokens, structure_tokens), dim=2)
        keys = self.key_projection(paired_tokens).reshape(
            batch_size, num_queries, 2, self.num_heads, self.head_dim
        )
        keys = keys.permute(0, 1, 3, 2, 4)
        structure_values = self.structure_value_projection(structure_tokens).reshape(
            batch_size, num_queries, self.num_heads, self.head_dim
        )
        return clip_tokens, queries, keys, structure_values

    def _forward_with_attention_weights(
        self,
        structure_features: torch.Tensor,
        clip_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        clip_tokens, queries, keys, structure_values = (
            self._project_attention_inputs(structure_features, clip_features)
        )
        attention_logits = torch.einsum(
            "bqhd,bqhsd->bqhs", queries, keys
        ) * (self.head_dim ** -0.5)
        attention_weights = F.softmax(attention_logits, dim=-1)
        dropped_weights = F.dropout(
            attention_weights,
            p=self.dropout,
            training=self.training,
        )
        correction_heads = dropped_weights[..., 1, None] * structure_values
        correction = correction_heads.reshape(
            clip_tokens.shape[0], clip_tokens.shape[1], self.embed_dim
        )
        output = self.output_norm(
            clip_tokens + self.output_projection(correction)
        )
        return output, attention_weights.unsqueeze(-2)

    def forward_with_attention_weights(
        self,
        structure_features: torch.Tensor,
        clip_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return output and gates shaped ``[B, Q, heads, 1, 2]``."""
        return self._forward_with_attention_weights(
            structure_features, clip_features
        )

    def projected_paired_values(
        self,
        structure_features: torch.Tensor,
        clip_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return diagnostic paired values with an explicit zero self value."""
        _, _, _, structure_values = self._project_attention_inputs(
            structure_features, clip_features
        )
        return torch.stack((torch.zeros_like(structure_values), structure_values), dim=-2)

    def forward(
        self,
        structure_features: torch.Tensor,
        clip_features: torch.Tensor,
    ) -> torch.Tensor:
        output, _ = self._forward_with_attention_weights(
            structure_features, clip_features
        )
        return output


LocalDiscriminativePoolingModule = Union[
    LocalDiscriminativePooling,
    PairedGatedCorrection,
]


def build_local_discriminative_pooling(
    *,
    ldp_type: str,
    structure_dim: int,
    embed_dim: int,
    num_heads: int,
    dropout: float,
) -> LocalDiscriminativePoolingModule:
    """Build one explicitly selected LDP topology."""
    normalized_type = ldp_type.lower()
    module_kwargs = {
        "structure_dim": structure_dim,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "dropout": dropout,
    }
    if normalized_type == "global":
        return LocalDiscriminativePooling(**module_kwargs)
    if normalized_type == "paired_gated_correction":
        return PairedGatedCorrection(**module_kwargs)
    raise ValueError(
        "Unsupported LDP type: "
        f"{ldp_type!r}; expected 'global' or 'paired_gated_correction'"
    )
