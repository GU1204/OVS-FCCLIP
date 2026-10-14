# FC-CLIP LDP 与 PGC-LDP 当前实现设计

## 目标和边界

当前实现提供两种 LDP topology：

```text
MODEL.FC_CLIP.LDP.TYPE: "global"
MODEL.FC_CLIP.LDP.TYPE: "paired_gated_correction"
```

两者共享同一条 FC-CLIP 集成路径：

- 只作用于 final decoder 输出；
- 只替换非 void 类别 logits；
- 原 FC-CLIP final classifier 保留 void/objectness logit；
- auxiliary decoder logits、mask loss、dice loss、Hungarian matcher 不变；
- CLIP image/text encoder 保持冻结；
- 不新增 LDP 专用 loss、temperature 或 text adapter。

当前代码不是严格 PosSAM 复现。结构分支来自 FC-CLIP pixel/transformer decoder，
语义分支来自同一个 frozen ConvNeXt CLIP 的 dense feature mask pooling；没有引入
SAM encoder、IoU head 或 MASE。

## 数据流

```text
image
  |
  v
Frozen ConvNeXt CLIP
  |-- res2/res3/res4/res5
  |       |
  |       v
  |   Pixel Decoder + Transformer Decoder
  |       |-- final pred_masks [B,Q,Hm,Wm]
  |       `-- final region_features [B,Q,256] --------+
  |                                                    |
  `-- clip_vis_dense [B,C,Hc,Wc]                       |
          |                                            |
          | final pred_masks                           |
          v                                            |
     hard detached mask pooling                        |
          |                                            |
     frozen CLIP visual projection                     |
          |                                            |
          v                                            v
     clip region tokens [B,Q,D] --------------> LDP / PGC
                                                       |
                                                       v
                                             fused region tokens [B,Q,D]
                                                       |
                                             CLIP text similarity
                                                       |
                                             non-void class logits

original FC-CLIP final classifier ----------------> base void logit

final logits = concat(ldp_or_pgc_logits[..., :-1], base_logits[..., -1:])
```

在当前 ConvNeXt-B WaterOVS 配置中：

| 符号 | 含义 | 形状 |
| --- | --- | --- |
| `Q` | object queries | 250 |
| `D` | CLIP text/image embedding dim | 640 |
| `region_features` | final decoder structure token | `[B,250,256]` |
| `clip region tokens` | mask-pooled CLIP semantic token | `[B,250,640]` |
| train logits | 84 个训练类 + void | `[B,250,85]` |
| val logits | 115 个验证类 + void | `[B,250,116]` |

## Structure token

Transformer decoder final prediction head 额外返回 `pred_region_features`：

```text
decoder_output = decoder_norm(output).transpose(0, 1)
mask_embed = mask_embed_head(decoder_output)
pred_masks = einsum(mask_embed, mask_features)
mask_region = MaskPooling(mask_features, pred_masks)
mask_region = mask_pooling_projection(mask_region)
region_features = mask_region + decoder_output
```

只有 final 层暴露该 feature。9 个 auxiliary 层继续使用原 FC-CLIP 分类路径。

## CLIP semantic token

`FCCLIP.forward()` 使用 final `pred_masks` 对 `features["clip_vis_dense"]` 做
mask pooling：

```text
pred_masks -> bilinear resize to clip_vis_dense resolution
clip_vis_dense + resized pred_masks -> MaskPooling
pooled token -> backbone.visual_prediction_forward
```

ConvNeXt 使用 pooled token 直接进入 frozen visual projection；ResNet 路径保留
mask 参数接口。未知 backbone 会抛出 `NotImplementedError`。

当前 `MaskPooling` 是 hard detached pooling：

```text
mask = mask.detach()
mask = (mask > 0).to(mask.dtype)
```

因此分类 loss 不通过 mask 权重直接反传到 `pred_masks`。soft differentiable pooling
不是当前实现的一部分，必须作为独立消融。

## Global LDP

实现类：

```text
fcclip/modeling/local_discriminative_pooling.py
LocalDiscriminativePooling
```

核心计算：

```text
S = Linear(256,D)(region_features)
C = Linear(D,D)(clip_tokens)

Q = C
K = concat(S, C, channel_dim)
V = S

delta = MultiheadAttention(Q, K, V)
output = LayerNorm(C + delta)
```

特点：

- `K` 在 channel 维拼接，token 数仍为 `Q=250`；
- 每个 CLIP query 可以读取所有 proposal 的 structure token；
- `clip_projection` 使用 identity weight + zero bias 初始化；
- `structure_projection` 使用 Xavier uniform 初始化；
- 参数量约 2,626,560。

## PGC-LDP

实现类：

```text
fcclip/modeling/local_discriminative_pooling.py
PairedGatedCorrection
```

PGC-LDP 的设计目标是限制 proposal 间错配：同一 index 的 CLIP token 和
structure token 配对，structure 只作为 correction 写入对应的 CLIP residual anchor。

核心计算：

```text
S_i = Linear(256,D)(region_features_i)
C_i = Linear(D,D)(clip_tokens_i)

q_i = W_q C_i
k_i = [W_k C_i, W_k S_i]
v_i = [0, W_v S_i]

a_i = softmax(q_i k_i^T / sqrt(head_dim))
correction_i = a_i[structure] * W_v S_i
output_i = LayerNorm(C_i + W_o correction_i)
```

关键约束：

- 每个 mask/head 只在 self key 和 paired structure key 两个候选之间选择；
- self candidate 的 value 是显式零张量，不经过带 bias 的 V projection；
- `a_i[structure]` 是写入 structure correction 的 gate；
- correction 为零时，输出仍是 `LayerNorm(C_i)`，不是 raw CLIP token 的逐元素恒等；
- 参数量约 2,216,960。

相对 global LDP，PGC 减少 proposal 间 attention score：

```text
global LDP: heads * Q * Q = 8 * 250 * 250 = 500000 / image
PGC-LDP:    heads * Q * 2 = 8 * 250 * 2   =   4000 / image
```

score tensor 缩小不等于端到端训练成本同比例下降，因为 projection、mask pooling、
decoder 和 loss 仍占主要开销。

## Preserve-void

`combine_ldp_non_void_logits()` 固定保留原 FC-CLIP void logit：

```python
torch.cat((ldp_logits[..., :-1], base_logits[..., -1:]), dim=-1)
```

这个行为没有 config switch。语义分工是：

- LDP/PGC 负责前景 query 的类别判别；
- 原 FC-CLIP classifier 负责 no-object/void 竞争；
- final CE 仍会训练原始 `class_embed` 的 void 路径；
- auxiliary CE 仍训练原始 `class_embed` 的所有辅助输出。

## 配置

LDP 默认关闭：

```yaml
MODEL:
  FC_CLIP:
    LDP:
      ENABLED: false
      TYPE: "global"
      NUM_HEADS: 8
      DROPOUT: 0.0
```

纯 LDP smoke：

```text
configs/coco/panoptic-segmentation/fcclip/
  fcclip_convnext_base_waterovs_id_ldp_noamp_3gpu_bs15_smoke_5k.yaml
```

PGC-LDP smoke：

```text
configs/coco/panoptic-segmentation/fcclip/
  fcclip_convnext_base_waterovs_id_ldp_paired_gated_correction_noamp_3gpu_bs15_smoke_5k.yaml
```

`*_preserve_void_*` 配置名保留是为了兼容早期实验命名；当前 preserve-void 已经是
LDP enabled 时的固定代码行为。

## 验证状态

迁移后已完成短程训练验证：

| topology | 日志 | loss 证据 |
| --- | --- | --- |
| global LDP | `output/verify_migration_ldp_3loss/log.txt` | iter 19: 64.73; iter 39: 44.49; iter 40: 44.01 |
| PGC-LDP | `output/verify_migration_pgc_3loss/log.txt` | iter 19: 62.78; iter 39: 42.97; iter 40: 42.57 |

该验证只证明迁移后的 LDP/PGC 路径能稳定启动并得到有限 loss，不证明完整 5k/50k
指标。后续 full run 需要单独记录 `实验记录.md` 并基于 `metrics.json` 评价。

## 局限

- 两路 token 同源于 CLIP，不能宣称复现 PosSAM 的 SAM+CLIP 异构互补；
- hard detached pooling 切断分类 loss 到 mask 权重的直接梯度；
- 当前只做 final-only LDP/PGC；
- preserve-void 是固定行为，不支持完整替换 void 的对照开关；
- 当前验证是 WaterOVS ID / ConvNeXt-B 迁移验证，不能外推到其他数据集或 backbone。
