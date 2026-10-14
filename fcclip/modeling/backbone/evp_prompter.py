from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F


EVPInputType = Literal[
    "identity",
    "fft",
    "all",
    "gaussian",
    "srm",
    "degradation",
]
EVPPromptType = Literal["highpass", "lowpass"]
EVPAdaptorType = Literal["adaptor", "fully_shared", "fully_unshared"]
EVPFusionType = Literal["add", "gated_add"]
EVPFrequencyRouterType = Literal[
    "none",
    "low_conditioned_gate",
]

HandcraftedPrompt = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
StagePrompt = Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]


def fft_visual_prompt(x: torch.Tensor, rate: float, prompt_type: EVPPromptType) -> torch.Tensor:
    if prompt_type not in ("highpass", "lowpass"):
        raise ValueError(f"Unknown EVP prompt_type: {prompt_type}")
    if not 0.0 < rate < 1.0:
        raise ValueError(f"EVP frequency rate must be in (0, 1), got {rate}")

    mask = torch.zeros_like(x)
    width, height = x.shape[-2:]
    line = int((width * height * rate) ** 0.5 // 2)
    mask[:, :, width // 2 - line : width // 2 + line, height // 2 - line : height // 2 + line] = 1

    fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
    if prompt_type == "highpass":
        fft = fft * (1 - mask)
    else:
        fft = fft * mask

    restored = torch.fft.ifft2(torch.fft.ifftshift(fft), norm="forward").real
    return restored.abs()


def degradation_visual_prompt(image: torch.Tensor, downsample_ratio: int = 16) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(f"degradation_visual_prompt expects [B, C, H, W], got {tuple(image.shape)}")
    if image.shape[1] != 3:
        raise ValueError(f"degradation_visual_prompt expects 3 image channels, got {image.shape[1]}")
    if downsample_ratio <= 0:
        raise ValueError(f"downsample_ratio must be positive, got {downsample_ratio}")

    height, width = image.shape[-2:]
    low_height = max(1, height // downsample_ratio)
    low_width = max(1, width // downsample_ratio)
    low_frequency = F.adaptive_avg_pool2d(image, output_size=(low_height, low_width))
    low_frequency = F.interpolate(low_frequency, size=(height, width), mode="bilinear", align_corners=False)

    channel_mean = image.mean(dim=(-2, -1), keepdim=True)
    gray_mean = channel_mean.mean(dim=1, keepdim=True)
    color_cast = channel_mean - gray_mean
    illumination_residual = low_frequency - channel_mean
    return illumination_residual + color_cast


class GaussianFilter(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor(
            [
                [1.0, 4.0, 6.0, 4.0, 1.0],
                [4.0, 16.0, 24.0, 16.0, 4.0],
                [6.0, 24.0, 36.0, 24.0, 6.0],
                [4.0, 16.0, 24.0, 16.0, 4.0],
                [1.0, 4.0, 6.0, 4.0, 1.0],
            ],
            dtype=torch.float32,
        )
        kernel = (kernel / 256.0).view(1, 1, 5, 5).repeat(3, 1, 1, 1)
        self.register_buffer("kernel", kernel, persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel.to(dtype=image.dtype, device=image.device)
        padded = F.pad(image, (2, 2, 2, 2), mode="reflect")
        return F.conv2d(padded, kernel, groups=image.shape[1])


class SRMFilter(nn.Module):
    def __init__(self):
        super().__init__()
        filter1 = [
            [0, 0, 0, 0, 0],
            [0, -1 / 4, 2 / 4, -1 / 4, 0],
            [0, 2 / 4, -4 / 4, 2 / 4, 0],
            [0, -1 / 4, 2 / 4, -1 / 4, 0],
            [0, 0, 0, 0, 0],
        ]
        filter2 = [
            [-1 / 12, 2 / 12, -2 / 12, 2 / 12, -1 / 12],
            [2 / 12, -6 / 12, 8 / 12, -6 / 12, 2 / 12],
            [-2 / 12, 8 / 12, -12 / 12, 8 / 12, -2 / 12],
            [2 / 12, -6 / 12, 8 / 12, -6 / 12, 2 / 12],
            [-1 / 12, 2 / 12, -2 / 12, 2 / 12, -1 / 12],
        ]
        filter3 = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1 / 2, -2 / 2, 1 / 2, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        weight = torch.tensor(
            [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]],
            dtype=torch.float32,
        )
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(dtype=image.dtype, device=image.device)
        return F.conv2d(image, weight, padding=2)


class OverlapPatchPromptEmbed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        self.proj = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.proj(x)
        spatial_shape = x.shape[-2:]
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        return self.norm(tokens), spatial_shape

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, (2.0 / fan_out) ** 0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class FrequencyResidualGate(nn.Module):
    def __init__(self, channels: int, output_bias: float = -2.0):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")

        gate_channels = 2 * channels + 1
        self.input_proj = nn.Conv2d(gate_channels, channels, kernel_size=1)
        self.depthwise_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.output_proj = nn.Conv2d(channels, 1, kernel_size=1)
        self.activation = nn.GELU()
        self.output_bias = output_bias
        self.reset_output_layer()

    def forward(self, f_clip: torch.Tensor, delta_freq: torch.Tensor) -> torch.Tensor:
        if f_clip.shape != delta_freq.shape:
            raise ValueError(
                f"f_clip and delta_freq must have the same shape, got {tuple(f_clip.shape)} and {tuple(delta_freq.shape)}"
            )
        if f_clip.ndim == 4:
            return self._forward_feature_map(f_clip, delta_freq)
        if f_clip.ndim == 3:
            f_clip_map = f_clip.transpose(1, 2).unsqueeze(-1).contiguous()
            delta_freq_map = delta_freq.transpose(1, 2).unsqueeze(-1).contiguous()
            gate = self._forward_feature_map(f_clip_map, delta_freq_map)
            return gate.squeeze(1).squeeze(-1).unsqueeze(-1)
        raise ValueError(f"FrequencyResidualGate expects [B, C, H, W] or [B, N, C], got {tuple(f_clip.shape)}")

    def reset_output_layer(self) -> None:
        nn.init.zeros_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.constant_(self.output_proj.bias, self.output_bias)

    def _forward_feature_map(self, f_clip: torch.Tensor, delta_freq: torch.Tensor) -> torch.Tensor:
        delta_norm = delta_freq.norm(p=2, dim=1, keepdim=True)
        gate_input = torch.cat([f_clip, delta_freq, delta_norm], dim=1)
        gate = self.input_proj(gate_input)
        gate = self.activation(gate)
        gate = self.depthwise_conv(gate)
        gate = self.activation(gate)
        return torch.sigmoid(self.output_proj(gate))


class LowFrequencyConditionedResidualGate(nn.Module):
    def __init__(self, channels: int, output_bias: float = -2.0):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")

        gate_channels = 3 * channels + 2
        self.input_proj = nn.Conv2d(gate_channels, channels, kernel_size=1)
        self.depthwise_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.output_proj = nn.Conv2d(channels, 1, kernel_size=1)
        self.activation = nn.GELU()
        self.output_bias = output_bias
        self.reset_output_layer()

    def forward(self, f_clip: torch.Tensor, delta_freq: torch.Tensor, low_freq_condition: torch.Tensor) -> torch.Tensor:
        if f_clip.shape != delta_freq.shape or f_clip.shape != low_freq_condition.shape:
            raise ValueError(
                "f_clip, delta_freq, and low_freq_condition must have the same shape, "
                f"got {tuple(f_clip.shape)}, {tuple(delta_freq.shape)}, and {tuple(low_freq_condition.shape)}"
            )
        if f_clip.ndim == 4:
            return self._forward_feature_map(f_clip, delta_freq, low_freq_condition)
        if f_clip.ndim == 3:
            f_clip_map = f_clip.transpose(1, 2).unsqueeze(-1).contiguous()
            delta_freq_map = delta_freq.transpose(1, 2).unsqueeze(-1).contiguous()
            low_freq_map = low_freq_condition.transpose(1, 2).unsqueeze(-1).contiguous()
            gate = self._forward_feature_map(f_clip_map, delta_freq_map, low_freq_map)
            return gate.squeeze(1).squeeze(-1).unsqueeze(-1)
        raise ValueError(
            f"LowFrequencyConditionedResidualGate expects [B, C, H, W] or [B, N, C], got {tuple(f_clip.shape)}"
        )

    def reset_output_layer(self) -> None:
        nn.init.zeros_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.constant_(self.output_proj.bias, self.output_bias)

    def _forward_feature_map(
        self,
        f_clip: torch.Tensor,
        delta_freq: torch.Tensor,
        low_freq_condition: torch.Tensor,
    ) -> torch.Tensor:
        delta_norm = delta_freq.norm(p=2, dim=1, keepdim=True)
        low_norm = low_freq_condition.norm(p=2, dim=1, keepdim=True)
        gate_input = torch.cat([f_clip, delta_freq, low_freq_condition, delta_norm, low_norm], dim=1)
        gate = self.input_proj(gate_input)
        gate = self.activation(gate)
        gate = self.depthwise_conv(gate)
        gate = self.activation(gate)
        return torch.sigmoid(self.output_proj(gate))


class ConvNeXtEVPPromptGenerator(nn.Module):
    """
    SegFormer-EVP style prompt generator for ConvNeXt CLIP stages.

    The generator keeps the original EVP structure: explicit handcrafted visual
    prompts, embedding prompts from current stage features, and per-depth
    adaptor projections before residual injection.
    """

    def __init__(
        self,
        stage_channels: Sequence[int],
        stage_depths: Sequence[int],
        scale_factor: int = 4,
        tuning_stage: str = "1234",
        input_type: EVPInputType = "fft",
        prompt_type: EVPPromptType = "highpass",
        freq_rate: float = 0.25,
        handcrafted_tune: bool = True,
        embedding_tune: bool = True,
        adaptor: EVPAdaptorType = "adaptor",
        fusion_type: EVPFusionType = "add",
        gate_bias: float = -2.0,
        frequency_router: EVPFrequencyRouterType = "none",
        pixel_mean: Sequence[float] = (122.7709383, 116.7460125, 104.09373615),
        pixel_std: Sequence[float] = (68.5005327, 66.6321579, 70.32316305),
        input_scale: float = 255.0,
    ):
        super().__init__()
        if len(stage_channels) != 4:
            raise ValueError(f"ConvNeXt EVP expects 4 stage channels, got {len(stage_channels)}")
        if len(stage_depths) != 4:
            raise ValueError(f"ConvNeXt EVP expects 4 stage depths, got {len(stage_depths)}")
        if scale_factor <= 0:
            raise ValueError(f"scale_factor must be positive, got {scale_factor}")
        if input_type not in (
            "identity",
            "fft",
            "all",
            "gaussian",
            "srm",
            "degradation",
        ):
            raise ValueError(f"Unknown EVP input_type: {input_type}")
        if prompt_type not in ("highpass", "lowpass"):
            raise ValueError(f"Unknown EVP prompt_type: {prompt_type}")
        if adaptor not in ("adaptor", "fully_shared", "fully_unshared"):
            raise ValueError(f"Unknown EVP adaptor: {adaptor}")
        if fusion_type not in ("add", "gated_add"):
            raise ValueError(f"Unknown EVP fusion_type: {fusion_type}")
        supported_frequency_routers = ("none", "low_conditioned_gate")
        if frequency_router not in supported_frequency_routers:
            raise ValueError(f"Unknown EVP frequency_router: {frequency_router}")
        if frequency_router != "none" and fusion_type != "gated_add":
            raise ValueError("EVP frequency_router currently requires fusion_type='gated_add'")
        if frequency_router != "none" and not handcrafted_tune:
            raise ValueError("EVP frequency_router requires handcrafted_tune=True")
        if not handcrafted_tune and not embedding_tune:
            raise ValueError("At least one EVP prompt source must be enabled")
        if len(pixel_mean) != 3 or len(pixel_std) != 3:
            raise ValueError("pixel_mean and pixel_std must each contain three RGB values")
        if any(float(value) <= 0.0 for value in pixel_std):
            raise ValueError(f"pixel_std values must be positive, got {list(pixel_std)}")
        if input_scale <= 0.0:
            raise ValueError(f"input_scale must be positive, got {input_scale}")

        self.stage_channels = list(stage_channels)
        self.stage_depths = list(stage_depths)
        self.scale_factor = scale_factor
        self.tuning_stage = tuning_stage
        self.input_type = input_type
        self.prompt_type = prompt_type
        self.freq_rate = freq_rate
        self.handcrafted_tune = handcrafted_tune
        self.embedding_tune = embedding_tune
        self.adaptor = adaptor
        self.fusion_type = fusion_type
        self.gate_bias = gate_bias
        self.frequency_router = frequency_router
        self.last_gate_map: Optional[torch.Tensor] = None
        self.last_low_frequency_feature: Optional[torch.Tensor] = None
        self.last_gate_means: Dict[Tuple[int, int], torch.Tensor] = {}
        self.input_scale = float(input_scale)
        self.register_buffer(
            "pixel_mean",
            torch.tensor(pixel_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(pixel_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        self.stage_prompt_dims = []
        for channel in self.stage_channels:
            if channel <= 0:
                raise ValueError(f"stage_channels must be positive, got {stage_channels}")
            if channel % scale_factor != 0:
                raise ValueError(f"stage channel must be divisible by scale_factor: {channel} vs {scale_factor}")
            self.stage_prompt_dims.append(channel // scale_factor)
        for depth in self.stage_depths:
            if depth <= 0:
                raise ValueError(f"stage_depths must be positive, got {stage_depths}")

        invalid_stages = sorted(set(tuning_stage) - set("1234"))
        if invalid_stages:
            raise ValueError(f"EVP tuning_stage must contain only 1, 2, 3, 4; got {invalid_stages}")
        if not tuning_stage:
            raise ValueError("EVP tuning_stage must not be empty")
        self.enabled_stage_indices = sorted({int(stage) - 1 for stage in tuning_stage})
        self.max_enabled_stage_index = max(self.enabled_stage_indices)

        if input_type == "gaussian":
            self.input_filter: Optional[nn.Module] = GaussianFilter()
        elif input_type == "srm":
            self.input_filter = SRMFilter()
        else:
            self.input_filter = None

        self.handcrafted_generators: Optional[nn.ModuleList]
        if handcrafted_tune:
            prompt_dims = self.stage_prompt_dims
            self.handcrafted_generators = nn.ModuleList(
                [
                    OverlapPatchPromptEmbed(3, prompt_dims[0], kernel_size=7, stride=4),
                    OverlapPatchPromptEmbed(prompt_dims[0], prompt_dims[1], kernel_size=3, stride=2),
                    OverlapPatchPromptEmbed(prompt_dims[1], prompt_dims[2], kernel_size=3, stride=2),
                    OverlapPatchPromptEmbed(prompt_dims[2], prompt_dims[3], kernel_size=3, stride=2),
                ]
            )
        else:
            self.handcrafted_generators = None

        self.embedding_generators: Optional[nn.ModuleList]
        if embedding_tune:
            self.embedding_generators = nn.ModuleList(
                [nn.Linear(channel, prompt_dim) for channel, prompt_dim in zip(self.stage_channels, self.stage_prompt_dims)]
            )
        else:
            self.embedding_generators = None

        if adaptor == "adaptor":
            self.lightweight_mlps = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            nn.Sequential(nn.Linear(prompt_dim, prompt_dim), nn.GELU())
                            for _ in range(depth)
                        ]
                    )
                    for prompt_dim, depth in zip(self.stage_prompt_dims, self.stage_depths)
                ]
            )
            self.shared_mlps = nn.ModuleList(
                [nn.Linear(prompt_dim, channel) for prompt_dim, channel in zip(self.stage_prompt_dims, self.stage_channels)]
            )
            self.fully_shared_mlps = None
            self.fully_unshared_mlps = None
        elif adaptor == "fully_shared":
            self.lightweight_mlps = None
            self.shared_mlps = None
            self.fully_shared_mlps = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(prompt_dim, prompt_dim), nn.GELU(), nn.Linear(prompt_dim, channel))
                    for prompt_dim, channel in zip(self.stage_prompt_dims, self.stage_channels)
                ]
            )
            self.fully_unshared_mlps = None
        else:
            self.lightweight_mlps = None
            self.shared_mlps = None
            self.fully_shared_mlps = None
            self.fully_unshared_mlps = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            nn.Sequential(nn.Linear(prompt_dim, prompt_dim), nn.GELU(), nn.Linear(prompt_dim, channel))
                            for _ in range(depth)
                        ]
                    )
                    for prompt_dim, channel, depth in zip(self.stage_prompt_dims, self.stage_channels, self.stage_depths)
                ]
            )

        self.frequency_residual_gates: Optional[nn.ModuleList]
        if fusion_type == "gated_add":
            gate_class = (
                LowFrequencyConditionedResidualGate
                if frequency_router == "low_conditioned_gate"
                else FrequencyResidualGate
            )
            self.frequency_residual_gates = nn.ModuleList(
                [gate_class(channel, output_bias=gate_bias) for channel in self.stage_channels]
            )
        else:
            self.frequency_residual_gates = None

        self.apply(self._init_weights)
        if self.frequency_residual_gates is not None:
            for gate in self.frequency_residual_gates:
                gate.reset_output_layer()
        self._freeze_disabled_stage_parameters()

    def init_handcrafted(self, image: torch.Tensor) -> List[Optional[HandcraftedPrompt]]:
        if not self.handcrafted_tune:
            return [None, None, None, None]
        if self.handcrafted_generators is None:
            raise RuntimeError("handcrafted_tune=True requires handcrafted_generators")

        handcrafted = self._init_handcrafted_from_image(self._filter_image(image))
        if self.frequency_router == "none":
            return handcrafted

        low_frequency_handcrafted = self._init_handcrafted_from_image(self._low_frequency_image(image))
        return [
            None if prompt is None else (prompt, low_frequency_prompt)
            for prompt, low_frequency_prompt in zip(handcrafted, low_frequency_handcrafted)
        ]

    def _init_handcrafted_from_image(self, prompt_image: torch.Tensor) -> List[Optional[torch.Tensor]]:
        if self.handcrafted_generators is None:
            raise RuntimeError("handcrafted_tune=True requires handcrafted_generators")

        handcrafted: List[Optional[torch.Tensor]] = []
        stage_feature = prompt_image
        for stage_index, generator in enumerate(self.handcrafted_generators):
            if stage_index > self.max_enabled_stage_index:
                handcrafted.append(None)
                continue
            tokens, spatial_shape = generator(stage_feature)
            handcrafted.append(tokens if self.stage_enabled(stage_index) else None)
            stage_feature = self._tokens_to_feature(tokens, spatial_shape)
        return handcrafted

    def init_stage_prompt(
        self,
        stage_feature: torch.Tensor,
        handcrafted_feature: Optional[HandcraftedPrompt],
        stage_index: int,
    ) -> StagePrompt:
        self._check_stage_index(stage_index)
        low_frequency_feature: Optional[torch.Tensor] = None
        if isinstance(handcrafted_feature, tuple):
            handcrafted_feature, low_frequency_feature = handcrafted_feature
        embedding_feature: Optional[torch.Tensor] = None
        if self.embedding_tune:
            if self.embedding_generators is None:
                raise RuntimeError("embedding_tune=True requires embedding_generators")
            embedding_feature = self.embedding_generators[stage_index](self._feature_to_tokens(stage_feature))
        if self.handcrafted_tune and handcrafted_feature is None:
            raise RuntimeError(f"Missing handcrafted EVP prompt for enabled stage {stage_index + 1}")
        if self.frequency_router != "none" and low_frequency_feature is None:
            raise RuntimeError(f"Missing low-frequency EVP router prompt for enabled stage {stage_index + 1}")
        return handcrafted_feature, embedding_feature, low_frequency_feature, None

    def add_prompt(
        self,
        stage_feature: torch.Tensor,
        prompt: StagePrompt,
        stage_index: int,
        depth_index: int,
    ) -> torch.Tensor:
        return stage_feature + self.prompt_residual(
            stage_feature,
            prompt,
            stage_index,
            depth_index,
        )

    def prompt_residual(
        self,
        stage_feature: torch.Tensor,
        prompt: StagePrompt,
        stage_index: int,
        depth_index: int,
    ) -> torch.Tensor:
        """Return the EVP residual injected into one ConvNeXt block."""
        self._check_stage_index(stage_index)
        self.last_low_frequency_feature = None
        if depth_index < 0 or depth_index >= self.stage_depths[stage_index]:
            raise ValueError(f"Invalid EVP depth index {depth_index} for stage {stage_index + 1}")

        feat: Optional[torch.Tensor] = None
        if self.handcrafted_tune:
            feat = prompt[0]
        if self.embedding_tune:
            embedding_feature = prompt[1]
            if embedding_feature is None:
                raise RuntimeError(f"Missing embedding EVP prompt for enabled stage {stage_index + 1}")
            feat = embedding_feature if feat is None else feat + embedding_feature
        if feat is None:
            raise RuntimeError("No EVP prompt feature was produced")

        tokens = self._feature_to_tokens(stage_feature)
        if feat.shape[:2] != tokens.shape[:2]:
            raise ValueError(
                f"EVP prompt tokens {feat.shape[:2]} do not align with stage tokens {tokens.shape[:2]}"
            )

        feat = self._adapt_prompt_tokens(feat, stage_index, depth_index)
        prompt_feature = self._tokens_to_feature(feat, stage_feature.shape[-2:]).to(dtype=stage_feature.dtype)
        if self.fusion_type == "add":
            return prompt_feature
        if self.fusion_type == "gated_add":
            if self.frequency_residual_gates is None:
                raise RuntimeError("gated_add fusion requires frequency_residual_gates")
            if self.frequency_router == "low_conditioned_gate":
                low_frequency_tokens = prompt[2]
                if low_frequency_tokens is None:
                    raise RuntimeError("low_conditioned_gate requires low-frequency prompt tokens")
                if low_frequency_tokens.shape[:2] != tokens.shape[:2]:
                    raise ValueError(
                        "Low-frequency EVP prompt tokens "
                        f"{low_frequency_tokens.shape[:2]} do not align with stage tokens {tokens.shape[:2]}"
                    )
                low_frequency_tokens = self._adapt_prompt_tokens(low_frequency_tokens, stage_index, depth_index)
                low_frequency_feature = self._tokens_to_feature(
                    low_frequency_tokens, stage_feature.shape[-2:]
                ).to(dtype=stage_feature.dtype)
                self.last_low_frequency_feature = low_frequency_feature.detach()
                gate = self.frequency_residual_gates[stage_index](
                    stage_feature,
                    prompt_feature,
                    low_frequency_feature,
                ).to(dtype=stage_feature.dtype)
            else:
                gate = self.frequency_residual_gates[stage_index](stage_feature, prompt_feature).to(
                    dtype=stage_feature.dtype
                )
            self.last_gate_map = gate.detach()
            self.last_gate_means[(stage_index, depth_index)] = gate.detach().mean()
            return gate * prompt_feature
        raise ValueError(f"Unknown EVP fusion_type: {self.fusion_type}")

    def _adapt_prompt_tokens(self, feat: torch.Tensor, stage_index: int, depth_index: int) -> torch.Tensor:
        if self.adaptor == "adaptor":
            if self.lightweight_mlps is None or self.shared_mlps is None:
                raise RuntimeError("adaptor mode requires lightweight_mlps and shared_mlps")
            feat = self.lightweight_mlps[stage_index][depth_index](feat)
            return self.shared_mlps[stage_index](feat)
        elif self.adaptor == "fully_shared":
            if self.fully_shared_mlps is None:
                raise RuntimeError("fully_shared mode requires fully_shared_mlps")
            return self.fully_shared_mlps[stage_index](feat)
        elif self.adaptor == "fully_unshared":
            if self.fully_unshared_mlps is None:
                raise RuntimeError("fully_unshared mode requires fully_unshared_mlps")
            return self.fully_unshared_mlps[stage_index][depth_index](feat)
        else:
            raise ValueError(f"Unknown EVP adaptor: {self.adaptor}")

    def _filter_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.input_type == "identity":
            return image
        if self.input_type == "fft":
            return fft_visual_prompt(image, self.freq_rate, self.prompt_type)
        if self.input_type == "all":
            return torch.zeros_like(image)
        if self.input_type == "degradation":
            return degradation_visual_prompt(image)
        if self.input_filter is None:
            raise RuntimeError(f"EVP input_type {self.input_type} requires an input_filter")
        return self.input_filter(image)

    def _normalized_to_rgb(self, image: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=image.device.type, enabled=False):
            image_float = image.float()
            image_rgb = image_float * self.pixel_std + self.pixel_mean
            return (image_rgb / self.input_scale).clamp(0.0, 1.0)

    def _low_frequency_image(self, image: torch.Tensor) -> torch.Tensor:
        return fft_visual_prompt(image, self.freq_rate, "lowpass")

    def stage_enabled(self, stage_index: int) -> bool:
        return str(stage_index + 1) in self.tuning_stage

    def _freeze_disabled_stage_parameters(self) -> None:
        for stage_index in range(4):
            if self.stage_enabled(stage_index):
                continue
            stage_modules: List[nn.Module] = []
            if self.handcrafted_generators is not None:
                stage_modules.append(self.handcrafted_generators[stage_index])
            if self.embedding_generators is not None:
                stage_modules.append(self.embedding_generators[stage_index])
            if self.lightweight_mlps is not None:
                stage_modules.append(self.lightweight_mlps[stage_index])
            if self.shared_mlps is not None:
                stage_modules.append(self.shared_mlps[stage_index])
            if self.fully_shared_mlps is not None:
                stage_modules.append(self.fully_shared_mlps[stage_index])
            if self.fully_unshared_mlps is not None:
                stage_modules.append(self.fully_unshared_mlps[stage_index])
            if self.frequency_residual_gates is not None:
                stage_modules.append(self.frequency_residual_gates[stage_index])
            for module in stage_modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    def _check_stage_index(self, stage_index: int) -> None:
        if stage_index < 0 or stage_index >= 4:
            raise ValueError(f"stage_index must be in [0, 3], got {stage_index}")
        if not self.stage_enabled(stage_index):
            raise ValueError(f"EVP stage {stage_index + 1} is not enabled by tuning_stage={self.tuning_stage}")

    @staticmethod
    def _feature_to_tokens(feature: torch.Tensor) -> torch.Tensor:
        return feature.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def _tokens_to_feature(tokens: torch.Tensor, spatial_shape: Tuple[int, int]) -> torch.Tensor:
        height, width = spatial_shape
        return tokens.transpose(1, 2).reshape(tokens.shape[0], -1, height, width).contiguous()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, (2.0 / fan_out) ** 0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
