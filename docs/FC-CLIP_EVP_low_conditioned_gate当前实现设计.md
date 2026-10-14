# FC-CLIP EVP low_conditioned_gate 当前实现设计

## 1. 文档定位

本文档描述 `/data/gjf/output/fc-clip` 中已经实现并完成 WaterOVS ID 训练验证的
**EVP low-conditioned gate**。当前主配置以 FFT high-pass prompt 作为实际写入
ConvNeXt 的视觉残差，以同图 FFT low-pass prompt 作为空间 gate 的条件：

> 高频分支提出“写入什么”，低频分支判断“在哪里写入”；低频条件本身不直接写回
> 冻结 CLIP backbone。

本文聚焦当前 `EVP_INPUT_TYPE=fft`、`EVP_PROMPT_TYPE=highpass`、
`EVP_FREQUENCY_ROUTER=low_conditioned_gate`、`EVP_TUNING_STAGE=12` 的实现。
它不是对所有 EVP 变体的统一说明，也不包含后续 FSP、多频带幅度通道调制或文本适配器。

截至 2026-08-18，当前证据状态为：

- low-conditioned gate 已集成到 ConvNeXt-B FC-CLIP；
- 形状、配置约束、high/low 双分支和 residual 等价关系已有定向单元测试；
- WaterOVS ID 5k smoke AP 为 `40.0132`；
- 原 global-batch-16 full run 的最佳 checkpoint 是 40k，AP 为 `48.5650`，50k final
  为 `47.6700`；
- 另有关闭 embedding prompt、global batch 15 的 highpass-only + low-conditioned gate
  50k 结果 `48.4061 AP`，但它不是原配置的严格重复实验。

## 2. 设计动机与边界

### 2.1 普通 high-pass gated-add 的局限

普通 EVP high-pass gated-add 先生成高频 prompt residual，再使用当前 CLIP feature 和
该 residual 预测空间 gate：

$$
R=G(F,\Delta_H)\odot\Delta_H
$$

其中 $F$ 是当前 ConvNeXt block 前的 feature，$\Delta_H$ 是适配后的高频 prompt。
该 gate 能根据语义 feature 和高频响应控制注入强度，但缺少显式的低频场景条件。
水下图像中的照明、色偏、浑浊度和大尺度背景通常更集中在低频结构中，因此当前设计把
low-pass prompt 加入 gate 的条件空间。

### 2.2 low_conditioned_gate 的约束

当前实现固定以下边界：

1. high-pass handcrafted prompt 是唯一的频率残差来源；
2. embedding prompt 可与 high-pass prompt 相加，但不进入 low-pass 条件分支；
3. low-pass prompt 只参与 gate 预测，不直接加到 backbone feature；
4. gate 是 `[B,1,H,W]` 的空间 gate，同一位置的所有通道共享 gate 值；
5. high/low prompt 使用同一 stage/block adaptor 权重映射到相同 channel 空间；
6. prompt 在每个启用 stage 的每个 block 前注入；
7. 原始 CLIP 参数保持冻结，EVP generator、adaptor 和 gate 为可训练部分；
8. `EVP_TUNING_STAGE=12` 表示 ConvNeXt stage 1/2，对外输出对应 FC-CLIP 的
   `res2/res3`。

它不是高低频双残差求和，也不是通道 gate。后续
`low_conditioned_fsp_gate` 和 `low_conditioned_band_amplitude_gate` 在此空间 gate
之上增加通道调制，不属于本文 baseline。

## 3. 整体数据流

```text
normalized image [B,3,H,W]
  |                                      |
  | FFT high-pass                        | FFT low-pass
  v                                      v
high-pass response                   low-pass response
  |                                      |
  | shared handcrafted prompt pyramid    | shared handcrafted prompt pyramid
  v                                      v
high prompt tokens H_s               low prompt tokens L_s
  |                                      |
  | + embedding prompt E_s               | no embedding prompt
  |                                      |
  | same stage/block adaptor              | same stage/block adaptor
  v                                      v
high delta Delta_H [B,C,Hs,Ws]       low condition C_L [B,C,Hs,Ws]
             \                            /
              \                          /
               v                        v
          spatial gate G(F, Delta_H, C_L) [B,1,Hs,Ws]
                            |
                            v
               residual = gate * high delta
                            |
                            v
                  F <- F + residual
                            |
                            v
                     ConvNeXt block
```

ConvNeXt-B 的 stage channels 为 `[128,256,512,1024]`。在当前 stage1/2 配置中，
实际启用的 stage prompt 维度由 `EVP_SCALE_FACTOR=4` 压缩为 `[32,64]`，再由 adaptor
投影回 `[128,256]`。

## 4. High/low 输入构造

### 4.1 FFT mask

`fft_visual_prompt(x, rate, prompt_type)` 对 `[B,C,H,W]` 输入做二维 FFT，并在移频后的
频谱中心构造方形 mask。当前 `EVP_FREQ_RATE=0.25`：

$$
\ell=\left\lfloor\frac{\sqrt{HW\cdot r}}{2}\right\rfloor
$$

中心区域近似覆盖频谱面积的 25%，但实际大小受整数取整和图像长宽比影响。

### 4.2 High-pass 主分支

当 `EVP_PROMPT_TYPE=highpass` 时，中心低频被清零，保留中心外的频率分量：

$$
I_H=\left|\operatorname{Re}\left(\mathcal{F}^{-1}((1-M)\odot\mathcal{F}(I))\right)\right|
$$

因此返回的是非负高频响应强度，而不是保留正负号的像素残差。$I_H$ 经四级
overlap patch embedding 构造 handcrafted prompt pyramid。

### 4.3 Low-pass 条件分支

只要 `EVP_FREQUENCY_ROUTER != none`，`init_handcrafted()` 就额外固定调用 low-pass：

$$
I_L=\left|\operatorname{Re}\left(\mathcal{F}^{-1}(M\odot\mathcal{F}(I))\right)\right|
$$

每个启用 stage 得到 `(high_prompt_tokens, low_prompt_tokens)`。low-pass 分支与主分支
使用同一 `EVP_FREQ_RATE` 和同一 handcrafted pyramid 拓扑，因此空间 token 能与
stage feature 对齐。

## 5. Stage prompt 与 per-block adaptor

### 5.1 Prompt 组成

对于 stage $s$，主分支 token 为：

$$
P_s^H=H_s+E_s
$$

其中 $H_s$ 是 high-pass handcrafted prompt，$E_s$ 是从当前 stage feature 经过线性层
生成的 embedding prompt。条件分支保持：

$$
P_s^L=L_s
$$

当前 baseline 不把 $E_s$ 加到 low-pass 分支。这样 low condition 保持为显式的图像
低频信号；`low_conditioned_shared_embedding_gate` 才是共享 embedding 的另一种设计。

### 5.2 Adaptor

默认 `EVP_ADAPTOR=adaptor` 时，每个 block 使用：

```text
per-block Linear(Ds -> Ds) + GELU
per-stage shared Linear(Ds -> Cs)
```

high/low 两路在同一个 block 调用同一个 adaptor：

$$
\Delta_{s,d}^H=A_{s,d}(P_s^H),\qquad C_{s,d}^L=A_{s,d}(P_s^L)
$$

共享参数使二者进入相同的 stage channel 空间，但两次前向的激活不同。对于 stage1/2，
每个 stage 内的所有 ConvNeXt block 都有独立的第一层轻量 MLP，回到 channel 维度的
第二层为 stage 共享。

## 6. LowFrequencyConditionedResidualGate

### 6.1 Gate 输入

对当前 block feature $F$、主残差 $\Delta_H$ 和低频条件 $C_L$，先计算逐位置通道范数：

$$
n_H=\|\Delta_H\|_2,\qquad n_L=\|C_L\|_2
$$

再拼接：

$$
X_G=[F;\Delta_H;C_L;n_H;n_L]
$$

若 feature channel 为 $C$，则 gate 输入为 `[B,3C+2,H,W]`。

### 6.2 Gate 网络

```text
Conv1x1(3C+2 -> C)
GELU
Depthwise Conv3x3(C -> C)
GELU
Conv1x1(C -> 1)
Sigmoid
```

记该网络为 $g_\theta$：

$$
G=\sigma(g_\theta(X_G)),\qquad G\in(0,1)^{B\times1\times H\times W}
$$

最终 residual 和 block 输入为：

$$
\boxed{R=G\odot\Delta_H}
$$

$$
\boxed{F'=F+R}
$$

low-pass 条件只通过 $G$ 间接影响输出；不存在 $F+C_L$ 或
$F+G\odot(\Delta_H+C_L)$ 路径。

### 6.3 初始化行为

gate 输出层初始化为零权重，bias 为 `EVP_GATE_BIAS=-2.0`：

$$
G_0=\sigma(-2)\approx0.1192
$$

训练开始时 gate 在所有样本和位置上相同，只写入约 11.9% 的 prompt residual。训练后
输出层权重才使 gate 依赖 $F$、$\Delta_H$ 和 $C_L$。这是一种保守残差初始化，但不是
严格 identity 初始化，因为初始 residual 不为零。

## 7. FC-CLIP 集成

### 7.1 构建与冻结边界

`fcclip/modeling/backbone/clip.py` 在 `EVP_ENABLED=true` 时构建
`ConvNeXtEVPPromptGenerator`。当前仅支持 ConvNeXt CLIP；非 ConvNeXt backbone 会直接
抛出 `ValueError`。

基础 CLIP 先由 `freeze_everything()` 冻结，随后创建的 EVP 参数保持可训练。因此训练
修改的是 prompt pyramid、embedding generator、adaptor 和 gate，而不是原始 CLIP
backbone 权重。

### 7.2 Block 接入点

每个 stage 先执行 downsample，再初始化一次 stage prompt；每个 block 前重新使用对应的
per-block adaptor 和 gate：

```text
x = stage.downsample(x)
prompt = init_stage_prompt(x, handcrafted_prompts[s], s)

for d, block in enumerate(stage.blocks):
    x = x + prompt_residual(x, prompt, s, d)
    x = block(x)
```

因此 stage prompt token 在 stage 内复用，但 residual 会随 block 当前 feature 和独立
adaptor/gate 重新计算。

### 7.3 对 FC-CLIP 下游的影响

本模块不直接修改 pixel decoder、transformer decoder、MaskPooling、文本 embedding、
分类公式或 void/objectness 路径。它通过改变冻结 CLIP ConvNeXt 的 `res2/res3` 特征，
间接影响 pixel decoder 输入以及后续 dense CLIP region classification。

## 8. 配置与使用

当前 Stage1/2 baseline 配置为：

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

示例配置：

```text
configs/coco/panoptic-segmentation/fcclip/
  fcclip_convnext_base_waterovs_id_evp_high_low_router_stage12_gated_add_gb2_smoke_5k.yaml
```

关键约束：

| 配置 | 当前约束 |
| --- | --- |
| `EVP_ENABLED` | 必须为 `true` 才构建 EVP。 |
| `EVP_FREQUENCY_ROUTER` | 设为 `low_conditioned_gate`。 |
| `EVP_FUSION_TYPE` | router 非 `none` 时必须为 `gated_add`。 |
| `EVP_HANDCRAFTED_TUNE` | 必须为 `true`，因为 low condition 来自 handcrafted pyramid。 |
| `EVP_EMBEDDING_TUNE` | 可开关；当前原始 baseline 为 `true`。 |
| `EVP_TUNING_STAGE` | 只能包含 `1/2/3/4` 且不能为空。 |
| `EVP_FREQ_RATE` | 必须位于 `(0,1)`。 |
| `EVP_SCALE_FACTOR` | 必须为正数，且各 stage channel 可整除。 |

非法组合会显式抛出 `ValueError` 或 `RuntimeError`，不会自动退回普通 gate。

## 9. 参数与计算特征

### 9.1 Gate 参数公式

单个 channel 为 $C$ 的 `LowFrequencyConditionedResidualGate` 参数量为：

$$
(3C+2)C+C+9C+C+C+1=3C^2+14C+1
$$

其中依次对应输入 `1x1` 卷积、depthwise `3x3` 卷积和输出 `1x1` 卷积及 bias。
stage1/2 的 $C=128/256$ 时，两级 low-conditioned gate 合计为 `251,138` 参数。

这不是完整 EVP 参数量；完整模块还包含 high/low 共用的 handcrafted pyramid、embedding
generator 和各 block adaptor。本文未重新执行端到端 FLOPs、吞吐或显存基准，因此不把
解析 gate 参数量外推成训练速度结论。

### 9.2 相对普通 gate 的新增输入

普通 `FrequencyResidualGate` 使用 `[F;\Delta_H;\|\Delta_H\|_2]`，输入 channel 为
`2C+1`。low-conditioned gate 新增 `C_L` 和 `\|C_L\|_2`，即首个 `1x1` 卷积增加
$C(C+1)$ 个权重；其余 gate 拓扑不变。此外，每个启用 stage/block 需要对 low tokens
额外执行一次共享 adaptor 前向。

## 10. 代码验证

对应实现的定向测试位于
`/data/gjf/code/fc-clip-amp/test/test_evp_frequency_router.py`，覆盖：

- high/low handcrafted prompt tuple 及相同 token 形状；
- gate map `[B,1,H,W]` 和 low-frequency condition `[B,C,H,W]`；
- `add_prompt(x,prompt) == x + prompt_residual(x,prompt)`；
- router 必须搭配 `gated_add`；
- router 必须启用 handcrafted prompt；
- 默认 `frequency_router=none` 不产生 low-frequency 条件槽位；
- odd/non-square 输入与 AMP forward/backward 等后续频率 router 共用回归检查。

历史实现验证记录中，`test/test_evp_frequency_router.py` 全部 `13 passed`。这是代码路径
验证，不等价于 full 训练收益证明。当前文档生成过程中未重新运行 GPU 测试或训练。

运行时可观察：

| 字段 | 含义 |
| --- | --- |
| `last_gate_map` | 最近一个 block 的空间 gate。 |
| `last_gate_means[(stage,depth)]` | 对应 block 的 gate 均值。 |
| `last_low_frequency_feature` | 最近一个 block 适配后的 low condition。 |

这些字段均保存 detached tensor，主要用于诊断，不构成训练 loss。

## 11. WaterOVS ID 实测结果

### 11.1 原 Stage1/2 配置的 5k smoke

| 实验 | AP | AP50 | AP75 | APs | APm | APl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| high-pass gated-add | 39.2826 | 48.0314 | 42.8567 | 0.0207 | 20.1593 | 40.9893 |
| low-conditioned gate | **40.0132** | **48.2474** | **43.0645** | **0.0649** | 16.8360 | **41.4728** |
| 差值 | **+0.7306** | **+0.2160** | **+0.2078** | **+0.0442** | -3.3233 | **+0.4835** |

该 5k 对照支持低频条件空间 gate 具有早期正向信号，但 APm 明显下降，且 smoke 不代表
full schedule。

### 11.2 原 Stage1/2 配置的 full 轨迹

| iterations | high-pass AP | low-conditioned AP | AP 差值 |
| ---: | ---: | ---: | ---: |
| 10k | 40.0229 | 43.5533 | +3.5304 |
| 20k | 43.5740 | 46.9754 | +3.4014 |
| 30k | 44.1438 | 47.0278 | +2.8840 |
| 40k | 45.2310 | **48.5650** | +3.3340 |
| 50k/final | 46.4874 | 47.6700 | +1.1826 |

low-conditioned run 的完整 final 指标为：

| checkpoint | AP | AP50 | AP75 | APs | APm | APl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 40k best | **48.5650** | **56.5632** | **52.3627** | **6.1033** | **29.6847** | **50.0924** |
| 50k final | 47.6700 | 55.2099 | 51.4246 | 3.2466 | 30.8491 | 49.2160 |

实测事实：

- low-conditioned gate 在 10k–50k 每个已记录 checkpoint 的 AP 都高于对应 high-pass run；
- 它在 40k 达到最佳 AP `48.5650`，50k 回落 `0.8950 AP`；
- final 相比 high-pass final 高 `1.1826 AP`；
- 40k 到 final 的 APm 上升，但 AP、AP50、AP75、APs 和 APl 均回落，因此不能只报告
  final 或只报告 best 来概括全部行为。

### 11.3 证据边界

上述原 Stage1/2 对照来自同一 WaterOVS ID 汇总口径，但目前只有单 seed。另一个
`highpass-only + low-conditioned gate` 以 FP32/global batch 15、关闭 embedding prompt
完成 50k，AP 为 `48.4061`；它证明该 router 在另一训练协议下也能稳定完成 full，
但不能与 global-batch-16 原 run 直接计算 embedding prompt 的因果增益。

## 12. 结果解释

### 12.1 已由实现和实验支持的内容

- low-pass feature 确实只作为 gate condition，实际写入 residual 仍是 high-pass 主分支；
- 空间 gate 能在真实 WaterOVS ID 训练中稳定前向、反向并完成 50k；
- 原配对轨迹中，low-conditioned gate 在所有已记录 checkpoint 均高于普通 high-pass
  gated-add；
- 最佳 checkpoint 为 40k 而非 final，存在训练后期回落；
- 当前收益并非各尺度一致：5k 的 APm 低于普通 high-pass，50k final 的 APs 也低于其
  自身 40k。

### 12.2 合理但尚未验证的解释

- low-pass 条件可能帮助 gate 区分真实目标边缘与水下噪声、散射或纹理伪影；
- 低频照明和颜色背景可能为高频残差提供更合适的空间选择信号；
- 40k 后回落可能来自 gate 饱和、浅层 prompt 过拟合或小目标响应退化。

这些机制目前缺少 gate 分布、目标/背景区域统计、频谱响应可视化和多 seed 支持，不能写成
已证实原因。

## 13. 已知限制

- FFT 使用中心方形 hard mask，不是径向或可学习频带；非方形输入的频率覆盖存在方向偏差；
- high/low IFFT 后取 `abs(real)`，会丢弃响应符号；
- `EVP_FREQ_RATE` 同时控制 high/low 划分，尚无独立 cutoff；
- gate 只有一个空间通道，不能对不同 feature channel 独立选择；
- high/low 共用 adaptor 有利于对齐，但也限制了两路各自学习独立映射；
- embedding prompt 只进入主分支，这一边界尚未通过 full 单变量消融确认；
- 初始 gate 约为 0.1192，不是严格零残差或 identity；
- 当前只验证 WaterOVS ID、ConvNeXt-B 和单 seed，不能外推到 CD、其他数据集或 backbone；
- 现有汇总未包含 gate 均值、方差、熵、目标/背景分组或 checkpoint 间漂移；
- 没有端到端 FLOPs、吞吐、显存和训练时间实测。

## 14. 后续验证优先级

建议保持一次只改一个变量：

1. 在相同代码、初始化、global batch、AMP 和 seed 下重跑 `router=none` 与
   `low_conditioned_gate`；
2. 增加至少 3 seeds，同时报告 best checkpoint 与 final 的均值和标准差；
3. 对 10k–50k 保存 per-stage/per-block gate 均值、方差、分位数和饱和比例；
4. 将 gate 与 GT/matched-mask 的前景、边界、背景区域对齐统计；
5. 单独消融 low condition、low norm、high norm，判断收益来自哪类输入；
6. 固定 router 后比较 `EVP_EMBEDDING_TUNE=true/false`；
7. 再比较方形 hard mask、平滑径向 mask 与 fixed dual-band，避免和自适应 selector 同时变化；
8. 完成 WaterOVS CD 和跨数据集验证后再讨论泛化。

## 15. 代码位置

| 文件 | 职责 |
| --- | --- |
| `fcclip/modeling/backbone/evp_prompter.py` | FFT prompt、双分支 pyramid、adaptor、low-conditioned gate 和 residual。 |
| `fcclip/modeling/backbone/clip.py` | 构建 EVP、冻结 CLIP、在 ConvNeXt stage/block 中注入 prompt。 |
| `fcclip/config.py` | 注册 `EVP_*` 默认配置与 router 选项。 |
| `/data/gjf/code/fc-clip-amp/test/test_evp_frequency_router.py` | 对应 high/low router 的形状、约束和 residual 测试。 |
| `configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_base_waterovs_id_evp_high_low_router_stage12_gated_add_gb2_smoke_5k.yaml` | 原 Stage1/2 low-conditioned gate 配置。 |
| `/data/gjf/files/exp/FC-CLIP整体实验文档.md` | full/checkpoint/smoke 汇总指标及证据边界。 |
| `/data/gjf/files/exp/FC-CLIP_all_experiment_results.csv` | 可追溯实验结果表。 |

## 16. 当前结论

当前 EVP low-conditioned gate 已实现清晰的“高频写入、低频条件选择”语义：high-pass
handcrafted prompt 与 embedding prompt 形成候选 residual，low-pass prompt 只进入空间
gate，最终以 `F + gate * high_delta` 的方式在每个 Stage1/2 ConvNeXt block 前注入。

实验上，原 Stage1/2 配对轨迹的 low-conditioned gate 在 5k–50k 所有已记录 checkpoint
均高于普通 high-pass gated-add；40k best 为 `48.5650 AP`，50k final 为 `47.6700 AP`，
final 对 high-pass final 的优势为 `+1.1826 AP`。这支持将它保留为有效的浅层视觉路由
baseline，但单 seed、后期回落和缺少 gate 机制统计意味着目前还不能把收益归因于某个
具体水下频率机制，也不能把 best checkpoint 当作稳定 final 表现。

## Material Passport

- Origin: 当前仓库源码、配置、测试及本地实验汇总
- Document Date: 2026-08-18
- Implementation Status: IMPLEMENTED
- Code Verification: historical directed tests `13 passed`; not rerun in this documentation turn
- WaterOVS ID Verification: 5k smoke and 50k full COMPLETED
- Evidence Scope: single-seed ID results; best and final reported separately
- External Retrieval: none
