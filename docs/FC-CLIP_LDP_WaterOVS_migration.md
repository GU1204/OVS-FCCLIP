# FC-CLIP LDP / PGC / EVP / WaterOVS 迁移总览

## 文档定位

本文档记录当前 `/data/gjf/output/fc-clip` 分支中已经迁移到干净 FC-CLIP
checkout 的代码范围、设计边界和验证状态。它不是实验结果汇总；AP、checkpoint
质量和长程训练结论必须以后续完整输出目录中的 `metrics.json`、`log.txt` 和
`实验记录.md` 为准。

当前仓库分支：

```text
feature/ldp-configs
```

当前迁移目标是把 FC-CLIP-AMP 中已经和 WaterOVS / LDP / PGC / 选定 EVP
路径有关的实现迁到干净 FC-CLIP，同时剔除未进入当前代码的 prompt/text-adapter
实验分支。

## 已迁移范围

| 模块 | 当前状态 | 主要文件 |
| --- | --- | --- |
| WaterOVS instance 注册 | 已迁移 | `fcclip/data/datasets/register_waterovs_instance.py` |
| WaterOVS base 配置 | 已迁移 | `configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_*_waterovs_id*.yaml` |
| Global LDP | 已迁移 | `fcclip/modeling/local_discriminative_pooling.py` |
| PGC-LDP | 已迁移 | `fcclip/modeling/local_discriminative_pooling.py` |
| Final decoder region feature 导出 | 已迁移 | `fcclip/modeling/transformer_decoder/fcclip_transformer_decoder.py` |
| LDP/PGC forward 集成 | 已迁移 | `fcclip/fcclip.py` |
| ConvNeXt EVP 基础路径 | 已迁移 | `fcclip/modeling/backbone/evp_prompter.py`, `fcclip/modeling/backbone/clip.py` |
| EVP low-conditioned gated-add | 已迁移 | `fcclip/modeling/backbone/evp_prompter.py` |
| open_clip `encode_text` 兼容 | 已迁移 | `fcclip/modeling/backbone/clip.py` |

当前 LDP/PGC 只替换 final decoder 输出的非 void 类别 logits。void/objectness
logit 始终保留原 FC-CLIP final classifier 路径，不再作为可开关实验项。

## 未迁移范围

以下源仓库设计和当前代码无直接对应实现，暂不迁移为当前设计文档：

| 候选 | 未迁移原因 |
| --- | --- |
| amplitude-selected bandpass EVP prompt | 当前 `EVP_INPUT_TYPE` 不支持自适应 bandpass selector |
| shared-embedding low gate | 当前 `EVP_FREQUENCY_ROUTER` 只支持 `none` 和 `low_conditioned_gate` |
| low-conditioned FSP gate | 当前未迁移 stage FFT channel modulator |
| low-conditioned band-amplitude gate | 当前未迁移多频带 log-amplitude 通道调制 |
| EVP text adapter / text geometry loss | 当前未迁移文本适配器、几何损失和 decoder/classifier 接入 |
| global amplitude / phase text adapter | 当前未迁移全局文本适配器和相关 loss |
| stage3/stage4 side adapter | 当前未迁移深层语义增强分支 |
| query/classification diagnostics | 当前未迁移诊断 evaluator 和输出记录逻辑 |

这些候选如果后续继续迁移，应作为独立任务选择，不应混入当前 LDP/PGC/EVP
low-conditioned 设计说明。

## 当前设计文档

当前代码对应的详细设计拆分为：

```text
docs/FC-CLIP_LDP_PGC当前实现设计.md
docs/FC-CLIP_EVP_low_conditioned_gate当前实现.md
docs/WaterOVS注册与训练配置说明.md
```

本文只保留迁移总览和跨模块边界。

## 配置入口

WaterOVS base：

```text
configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_waterovs_id.yaml
configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_base_waterovs_id.yaml
```

纯 LDP：

```text
configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_base_waterovs_id_ldp_noamp_3gpu_bs15_full.yaml
configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_base_waterovs_id_ldp_noamp_3gpu_bs15_smoke_5k.yaml
```

兼容保留的 preserve-void 命名配置：

```text
configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_base_waterovs_id_ldp_preserve_void_noamp_3gpu_bs15_full.yaml
configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_base_waterovs_id_ldp_preserve_void_noamp_3gpu_bs15_smoke_5k.yaml
```

PGC-LDP：

```text
configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_base_waterovs_id_ldp_paired_gated_correction_noamp_3gpu_bs15_smoke_5k.yaml
```

EVP low-conditioned gated-add：

```text
configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_base_waterovs_id_evp_high_low_router_stage12_gated_add_gb2_smoke_5k.yaml
```

## 迁移后短程验证

已完成三条 3 GPU / global batch 15 的短程训练验证，每条运行到至少 3 条
Detectron2 loss 记录。该验证只说明模型可构建、可前向/反向、loss 为有限值、短程训练
没有立即出现 OOM/traceback；不构成 AP 或收敛质量结论。

| 路径 | 配置 | 证据日志 | loss 记录 |
| --- | --- | --- | --- |
| 纯 LDP | `fcclip_convnext_base_waterovs_id_ldp_preserve_void_noamp_3gpu_bs15_smoke_5k.yaml` | `output/verify_migration_ldp_3loss/log.txt` | iter 19: 64.73; iter 39: 44.49; iter 40: 44.01 |
| 纯 PGC-LDP | `fcclip_convnext_base_waterovs_id_ldp_paired_gated_correction_noamp_3gpu_bs15_smoke_5k.yaml` | `output/verify_migration_pgc_3loss/log.txt` | iter 19: 62.78; iter 39: 42.97; iter 40: 42.57 |
| 纯 EVP | `fcclip_convnext_base_waterovs_id_evp_high_low_router_stage12_gated_add_gb2_smoke_5k.yaml` | `output/verify_migration_evp_3loss/log.txt` | iter 19: 171.1; iter 39: 50.01; iter 40: 49.09 |

`output/verify_migration_*/` 已在 `.gitignore` 中忽略。实验输出、checkpoint、
日志和 evaluator 中间状态不应纳入 Git。

## 当前边界

- LDP/PGC 与 EVP 是独立路径；当前没有组合 LDP+EVP 的迁移配置。
- PGC 的 `TYPE: paired_gated_correction` 只改变 LDP topology，不改变
  WaterOVS 数据、matcher、loss 权重或 open-vocabulary 几何融合逻辑。
- `EVP_TUNING_STAGE: 12` 表示 ConvNeXt stage 1 和 stage 2，即 `res2/res3`，
  不是第 12 个 stage。
- `SOLVER.AMP.ENABLED: false` 是训练自动混合精度开关；不要和历史
  amplitude/text adapter 实验中的 AMP 字样混淆。
- 当前文档不把历史 FC-CLIP-AMP 中的 AP 结果迁移为本仓库实测结果。
