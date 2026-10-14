# FC-CLIP EVP low-conditioned gated-add 当前实现

## 文档定位

本文只描述当前 `/data/gjf/output/fc-clip` 中已经迁移的 EVP 子集：

```yaml
MODEL:
  FC_CLIP:
    EVP_ENABLED: true
    EVP_INPUT_TYPE: "fft"
    EVP_PROMPT_TYPE: "highpass"
    EVP_FREQ_RATE: 0.25
    EVP_TUNING_STAGE: 12
    EVP_SCALE_FACTOR: 4
    EVP_HANDCRAFTED_TUNE: true
    EVP_EMBEDDING_TUNE: true
    EVP_ADAPTOR: "adaptor"
    EVP_FUSION_TYPE: "gated_add"
    EVP_GATE_BIAS: -2.0
    EVP_FREQUENCY_ROUTER: "low_conditioned_gate"
```

当前实现只支持 ConvNeXt CLIP backbone。非 ConvNeXt backbone 开启
`EVP_ENABLED=true` 会在构造阶段报错。

## 代码入口

| 文件 | 职责 |
| --- | --- |
| `fcclip/config.py` | 注册 EVP 默认配置 |
| `fcclip/modeling/backbone/clip.py` | 构建 `ConvNeXtEVPPromptGenerator`，在 ConvNeXt stage/block 中注入 residual |
| `fcclip/modeling/backbone/evp_prompter.py` | FFT prompt、prompt adaptor、gated-add、low-conditioned gate |
| `configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_base_waterovs_id_evp_high_low_router_stage12_gated_add_gb2_smoke_5k.yaml` | 当前 smoke 配置 |

## 总体数据流

当前配置的核心思想是：

- FFT high-pass 图像生成实际注入 ConvNeXt feature 的 prompt residual；
- FFT low-pass 图像只生成 gate 条件，不直接写回 backbone；
- embedding prompt 只叠加到 high-pass 注入分支；
- high-pass 和 low-pass prompt 复用同一套 stage/block adaptor，被映射到同一
  stage channel 空间后再门控。

```text
input image
  |-- FFT high-pass -> handcrafted prompt pyramid -> high prompt tokens
  |-- FFT low-pass  -> handcrafted prompt pyramid -> low condition tokens
  `-- ConvNeXt stage feature -> embedding prompt tokens

enabled ConvNeXt block:
  high prompt + embedding prompt -> adaptor -> high_delta [B,C,H,W]
  low condition prompt           -> adaptor -> low_condition [B,C,H,W]
  gate = G(stage_feature, high_delta, low_condition)
  residual = gate * high_delta
  stage_feature = block(stage_feature + residual)
```

## FFT prompt

`fft_visual_prompt(image, rate, prompt_type)` 支持：

```text
prompt_type: "highpass" | "lowpass"
0.0 < rate < 1.0
image shape: [B,C,H,W]
```

当前 smoke 配置使用：

```text
EVP_INPUT_TYPE: "fft"
EVP_PROMPT_TYPE: "highpass"
EVP_FREQ_RATE: 0.25
```

high-pass 分支保留中心频率 mask 外的分量，low-conditioned gate 额外调用
low-pass 分支作为 gate 条件。`EVP_PROMPT_TYPE` 仍然保持 `"highpass"`，不需要把
主分支改成 `"lowpass"`。

## Stage 选择

`EVP_TUNING_STAGE` 被转成字符串解析，只允许包含 `1,2,3,4`：

```text
"12" -> stage 1 和 stage 2
```

对 ConvNeXt 输出命名来说，stage 1/2 对应 `res2/res3`。这里的 `12` 不是“第十二
阶段”。

未启用 stage 的 EVP 参数会被冻结，stage forward 走原始 ConvNeXt stage。

## Prompt adaptor

默认 `EVP_ADAPTOR: "adaptor"`：

```text
per-block Linear(prompt_dim, prompt_dim) + GELU
per-stage shared Linear(prompt_dim, channel)
```

当前也保留 `fully_shared` 和 `fully_unshared` 两种 adaptor 选项，但当前 smoke
配置只使用 `adaptor`。

## LowFrequencyConditionedResidualGate

`EVP_FUSION_TYPE: "gated_add"` 且
`EVP_FREQUENCY_ROUTER: "low_conditioned_gate"` 时，gate 类选择为
`LowFrequencyConditionedResidualGate`。

输入：

```text
f_clip:             当前 ConvNeXt block 前 feature, [B,C,H,W]
delta_freq:         high-pass + embedding 得到的 residual, [B,C,H,W]
low_freq_condition: low-pass prompt 得到的条件 feature, [B,C,H,W]
```

gate 输入拼接：

```text
delta_norm = ||delta_freq||_2 over channel
low_norm   = ||low_freq_condition||_2 over channel

gate_input = concat(
  f_clip,
  delta_freq,
  low_freq_condition,
  delta_norm,
  low_norm
)  # [B, 3C+2, H, W]
```

gate 网络：

```text
Conv1x1(3C+2 -> C)
GELU
Depthwise Conv3x3(C -> C)
GELU
Conv1x1(C -> 1)
Sigmoid
```

输出：

```text
residual = gate * high_delta
gate shape = [B,1,H,W]
```

gate 是空间 gate，不是通道 gate；同一空间位置上所有 channel 共享一个 gate 值。

## 初始化

gate 输出层初始化为：

```text
output_proj.weight = 0
output_proj.bias = EVP_GATE_BIAS
```

默认 `EVP_GATE_BIAS=-2.0`，训练开始时：

```text
sigmoid(-2.0) ~= 0.119
```

因此初始 residual 注入较保守，且 gate 初始与输入内容无关；训练后输出层权重学习
到非零值，gate 才依赖 `f_clip`、`high_delta` 和 `low_condition`。

## 配置约束

构造阶段会显式检查：

| 约束 | 失败行为 |
| --- | --- |
| 只支持 ConvNeXt backbone | `ValueError` |
| `EVP_FREQUENCY_ROUTER!="none"` 要求 `EVP_FUSION_TYPE="gated_add"` | `ValueError` |
| `EVP_FREQUENCY_ROUTER!="none"` 要求 `EVP_HANDCRAFTED_TUNE=true` | `ValueError` |
| `EVP_HANDCRAFTED_TUNE` 和 `EVP_EMBEDDING_TUNE` 不能同时为 `false` | `ValueError` |
| `EVP_TUNING_STAGE` 只能包含 `1,2,3,4` 且不能为空 | `ValueError` |
| `EVP_FREQ_RATE` 必须在 `(0,1)` | `ValueError` |

当前没有 silent fallback。

## 未迁移的 EVP 变体

当前代码不支持以下 router/input/text 变体：

```text
amplitude_selected_bandpass
low_conditioned_shared_embedding_gate
low_conditioned_fsp_gate
low_conditioned_band_amplitude_gate
EVP_TEXT_ADAPTER
GLOBAL_AMPLITUDE_TEXT_ADAPTER
GLOBAL_PHASE_TEXT_ADAPTER
```

这些设计不要写进当前 EVP low-conditioned gated-add 的实验结论。

## 验证状态

迁移后已完成纯 EVP 短程训练验证：

```text
日志: output/verify_migration_evp_3loss/log.txt
iter 19 total_loss: 171.1
iter 39 total_loss: 50.01
iter 40 total_loss: 49.09
```

该验证只说明当前 EVP low-conditioned gated-add 路径能构建、前向/反向并产生有限
loss，不说明 AP 或完整 5k/50k 收敛质量。
