# FC-CLIP Paired Gated Correction LDP 当前实现设计

## 1. 文档定位

本文档描述 `/data/gjf/code/fc-clip-amp` 中已经实现并完成实际训练验证的
**Paired Gated Correction LDP（PGC-LDP）**。它是原 global LDP 的 mask-wise
替代拓扑，核心目标是：

> 以同一 mask 的 CLIP region feature 作为语义主干，只允许配对的 decoder structure
> feature 产生可门控 correction，并继续由原 FC-CLIP 路径负责 void/objectness。

这里的 residual anchor 是可训练 `clip_projection(G_clip)`，不是 raw CLIP feature 的
严格恒等副本；即使 correction 为零，输出仍会经过 LayerNorm。本文用“保持”表示
“不给 residual 额外写入 value”，不表示最终输出逐元素等于输入。

本文是当前 PGC-LDP 代码、配置、计算量和实验状态的聚焦说明。候选结构的前期比较与
推导见 `docs/FC-CLIP_LDP_CrossViT与MaskWise改进评估.md`；原 global LDP 的完整
说明见 `docs/FC-CLIP_LDP当前实现设计.md`。

截至 2026-08-12，当前证据状态为：

- PGC-LDP 已实现并通过单元测试；
- 纯 PGC-LDP 已完成从 5k 到 50k 的 WaterOVS ID 训练；
- `PGC-LDP + EVP stage1/2` 已完成 10k、20k、30k 评估，40k checkpoint 已保存，
  本文撰写时 50k full 尚未完成；
- pure PGC 和 PGC+EVP 的结果必须分开解释，不能把 EVP 的增益归因于 PGC 本身。

## 2. 设计动机与边界

### 2.1 原 global LDP 的问题

原 LDP 使用 250 个 proposal 之间的全局 attention：

$$
S=W_sG_{structure},\qquad C=W_cG_{clip}
$$

$$
Q=C,\qquad K=[S;C]_{channel},\qquad V=S
$$

因此 query $i$ 可以读取任意 structure token $S_j$。该设计能够提供 proposal 间上下文，
但也允许 mask $i$ 的语义表示写入 mask $j$ 的结构信息，存在 mask/category 对齐被破坏的
风险。decoder 已经包含 query self-attention，全局 LDP 是否还需要再次进行 proposal
mixing 也没有直接证据。

### 2.2 PGC-LDP 的约束

PGC-LDP 固定以下语义边界：

1. `G_clip[i]` 经 trainable `clip_projection` 后成为 mask $i$ 的 receiver 和 residual
   semantic anchor；
2. `G_structure[i]` 只能修正相同 index 的投影后 CLIP token；
3. 模型必须能够选择“不注入 structure correction”；
4. PGC 只替换 final non-void 类别 logits；
5. void logit、auxiliary decoder losses、mask/dice losses 和开放词汇融合保持原路径；
6. 第一版保持 full dimension 640，不同时引入 bottleneck、soft pooling 或新 loss。

它不是双向 CrossViT，也不是标准 `K=V=[C,S]` 的 MPCA。PGC 只更新 CLIP receiver，
并将 self candidate 定义为“不给 residual 写入 correction”。

## 3. 整体数据流

```text
image
  |
  v
Frozen ConvNeXt CLIP
  |-- res2/res3/res4/res5
  |       |
  |       v
  |   Pixel Decoder + Transformer Decoder
  |       |-- final pred_masks [B,250,Hm,Wm]
  |       `-- final G_structure [B,250,256] -------------+
  |                                                       |
  `-- clip_vis_dense [B,1024,Hc,Wc]                       |
          |                                                |
          | final pred_masks                               |
          v                                                |
     hard detached mask pooling                            |
          |                                                |
     Frozen CLIP visual projection                         |
          |                                                |
          v                                                v
       G_clip [B,250,640] --------------------------> PGC-LDP
                                                           |
                                                           v
                                                  G_pgc [B,250,640]
                                                           |
                                                CLIP text similarity
                                                           |
                                             PGC non-void class logits

Original FC-CLIP final classifier --------------------> base void logit

final logits = concat(PGC non-void logits, base void logit)
```

当前 ConvNeXt-B WaterOVS 设置中的主要形状为：

| 符号 | 含义 | 形状 |
| --- | --- | --- |
| `B` | batch size | 训练时每卡为 5 |
| `Q` | object queries | 250 |
| `G_structure` | final decoder region feature | `[B,250,256]` |
| `G_clip` | mask-pooled CLIP region feature | `[B,250,640]` |
| `G_pgc` | PGC 输出 | `[B,250,640]` |
| heads | attention heads | 8 |
| head dim | 每个 head 的维度 | 80 |
| train logits | 84 个训练类 + void | `[B,250,85]` |
| WaterOVS val logits | 115 个测试类 + void | `[B,250,116]` |

## 4. 输入构造与配对依据

### 4.1 Structure token

`G_structure` 来自 transformer decoder 最后一层的 `region_features`。当前路径为：

```text
decoder_output = decoder_norm(output).transpose(0, 1)
mask_embed = mask_embed_head(decoder_output)
pred_masks = einsum(mask_embed, mask_features)
mask_region = MaskPooling(mask_features, pred_masks)
mask_region = mask_pooling_projection(mask_region)
G_structure = mask_region + decoder_output
```

它融合了同一 query 的 decoder 表示和预测 mask 内的 pixel-decoder feature，形状为
`[B,250,256]`。

只有 final decoder 层返回 `pred_region_features`。9 个 auxiliary 层继续使用原
FC-CLIP 分类路径，不重复执行 PGC。

### 4.2 CLIP token

final `pred_masks` 通过 bilinear interpolation（`align_corners=False`）resize 到
`clip_vis_dense` 的空间分辨率，再经过 FC-CLIP 已有的 `MaskPooling` 和 frozen CLIP
visual projection：

```text
clip_vis_dense [B,1024,Hc,Wc]
  + pred_masks [B,250,Hm,Wm]
  -> hard mask pooling [B,250,1024]
  -> frozen visual projection
  -> G_clip [B,250,640]
```

### 4.3 为什么可以按 index 配对

配对不是因为两个 tensor 恰好都有 250 个 token，而是因为：

- `G_structure[i]` 来自 decoder query $i$ 及其预测 mask；
- `G_clip[i]` 使用同一个 `pred_masks[:,i]` 对 dense CLIP feature 做 pooling；
- 两路 feature 在进入 PGC 前没有重新排序。

因此 $S_i$ 与 $C_i$ 是同一个 proposal/mask 的结构和语义表示。

### 4.4 Hard detached pooling

当前 `MaskPooling` 在 `torch.no_grad()` 中执行：

```python
mask = mask.detach()
mask = (mask > 0).to(mask.dtype)
denorm = mask.sum((-1, -2), keepdim=True) + 1e-8
pooled = einsum("bchw,bqhw->bqc", feature, mask / denorm)
```

这意味着 PGC 分类 loss 不通过 pooling 权重直接回传到 mask logits。mask 仍由原有
mask/dice loss 和共享 decoder 表示训练。soft differentiable pooling 不属于当前
PGC-LDP，必须作为独立消融。

## 5. PGC-LDP 核心算法

模块位于：

```text
fcclip/modeling/local_discriminative_pooling.py
```

实现类为：

```python
class PairedGatedCorrection(nn.Module):
    ...
```

### 5.1 输入投影

先把两路输入映射到共同的 640 维空间：

$$
S_i=W_sG^i_{structure}+b_s
$$

$$
C_i=W_cG^i_{clip}+b_c
$$

- `structure_projection` 为 `Linear(256,640)`，Xavier uniform 初始化；
- `clip_projection` 为 `Linear(640,640)`，identity weight + zero bias 初始化；
- residual 使用投影后的 $C_i$，不是未经投影的 raw `G_clip`；
- 两个 projection 都可训练。

### 5.2 Paired Q/K

对于每个 mask $i$：

$$
q_i=W_qC_i+b_q
$$

$$
k_i^C=W_kC_i+b_k,\qquad k_i^S=W_kS_i+b_k
$$

$$
K_i=[k_i^C;k_i^S]_{token}
$$

同一个 `key_projection` 作用于 CLIP 和 structure token。reshape 后：

```text
Q: [B,Q,h,d_h]
K: [B,Q,h,2,d_h]
```

每个 mask、每个 head 只在 self 与 paired structure 两个 key 之间做选择，不读取其他
proposal。

### 5.3 Correction-only V

PGC 只对 structure token 计算 value：

$$
v_i^S=W_vS_i+b_v
$$

然后在 projection **之后**显式构造：

$$
V_i=[0;v_i^S]_{token}
$$

代码没有把零 token 送进带 bias 的 V projection。self value 由
`torch.zeros_like(structure_values)` 直接创建，因此即使 $b_v\neq0$，也仍有：

$$
V_i^{self}=0
$$

当前 `structure_value_projection` 保留 `bias=True` 是安全的；只有在先构造
`[0,S]`、再统一做 V projection 时，才必须设置 `bias=False` 才能保证零 self value。

### 5.4 Per-head keep-or-correct gate

每个 head 的 attention 为：

$$
[a^C_{i,h},a^S_{i,h}]
=\operatorname{Softmax}\left(
\frac{q_{i,h}[k^C_{i,h};k^S_{i,h}]^T}{\sqrt{d_h}}
\right)
$$

由于 self value 为零，head 输出化简为：

$$
R_{i,h}
=a^C_{i,h}\cdot0+a^S_{i,h}v^S_{i,h}
=a^S_{i,h}v^S_{i,h}
$$

因此 $a^S_{i,h}$ 是 V/O projection 和 LayerNorm 之前的 structure correction gate：

- $a^C\rightarrow1$：选择 self key，correction 接近零；
- $a^S\rightarrow1$：选择 structure key，使该 head 的 pre-output structure contribution
  接近 $v^S_{i,h}$；
- 中间值：按比例写入 structure correction。

self key 虽然对应零 value，但仍参与 softmax 分母，因此会影响 $a^S$ 并获得梯度。

### 5.5 输出投影、残差和归一化

将所有 head 的 correction 拼回 640 维：

$$
R_i=\operatorname{Concat}_h(R_{i,h})
$$

最终输出为：

$$
\boxed{
G^i_{pgc}=\operatorname{LayerNorm}\left(C_i+W_oR_i+b_o\right)
}
$$

这里“保持 semantic anchor”指 correction 为零时不向 residual 写入额外 value；严格地说，
输出仍为 `LayerNorm(C_i)`，不是逐元素恒等于原始 `G_clip[i]`。

### 5.6 为什么不使用其他 V 设计

标准 MPCA 若使用：

$$
V_i=[V_c(C_i);V_s(S_i)]
$$

则选择 self 仍会额外写入一份经过 V/O projection 的 CLIP feature：

$$
C_i+W_oV_c(C_i)
$$

此时 self 权重不再表示“保持”，并可能旋转 frozen CLIP 与 text embedding 的原始对齐。

若只使用单个 key：

$$
Q=C,\qquad K=S,\qquad V=S
$$

则单元素 softmax 恒为 1，structure correction 会被无条件写入，不再具有 gate。

## 6. FC-CLIP 集成

### 6.1 工厂与兼容性

LDP 由 `build_local_discriminative_pooling` 根据单一类型字段构建：

```yaml
MODEL:
  FC_CLIP:
    LDP:
      ENABLED: true
      TYPE: "paired_gated_correction"
      NUM_HEADS: 8
      DROPOUT: 0.0
```

默认值仍为 `TYPE: "global"`，因此已有 global LDP 配置不会被静默切换。LDP disabled
时不构建任何 PGC 参数，原模型路径保持不变。

### 6.2 Final-only classification

PGC 只作用于 final decoder 输出：

```text
final decoder/mask features -> PGC feature -> CLIP text similarity -> non-void logits
final void logit      -> original FC-CLIP classifier
aux logits x 9        -> original FC-CLIP classifier
mask/dice losses      -> unchanged
```

不新增 PGC 专用 loss、temperature 或 matcher 规则。

### 6.3 Preserve-void

PGC 复用原 `get_classification_logits` 接口和包含 void embedding 的 text classifier，
所以先计算完整类别 logits，再丢弃它产生的 void 列：

```python
final_logits = torch.cat(
    [pgc_logits[..., :-1], base_logits[..., -1:]],
    dim=-1,
)
```

语义分工为：

- PGC 决定前景 query 属于哪个类别；
- 原 FC-CLIP final classifier 保留 no-object/void 竞争；
- preserve-void 不是冻结 void，final CE 仍会训练 base void 路径。

这一约束来自已有失败证据：早期 LDP 同时替换 non-void 和 void 时，5k 只有
`1.2070 AP`；保留原 FC-CLIP void 后恢复到 `42.4391 AP`。

### 6.4 推理与开放词汇融合

推理继续保留两路分类：

```text
in-vocabulary:  PGC non-void + base void
out-of-vocabulary: G_clip + frozen CLIP text similarity
```

这里的 `G_clip` 指 frozen visual projection 之后、trainable `clip_projection` 之前的
mask-level feature，不是 dense CLIP feature map。PGC 前向已经计算的 `G_clip` 会被
out-of-vocabulary 分支复用，不重复执行 mask pooling 和 visual projection。

去掉 void 后，in-vocabulary 和 CLIP 分支分别做类别 softmax。当前几何融合为：

$$
p_k\propto
\begin{cases}
p_{in,k}^{1-\alpha}p_{clip,k}^{\alpha}, & k\in seen,\ \alpha=0.4\\
p_{in,k}^{1-\beta}p_{clip,k}^{\beta}, & k\in unseen,\ \beta=0.8
\end{cases}
$$

归一化 non-void 概率后，再乘以 $(1-p_{void})$ 并附加 base void probability。PGC
不会独占 unseen classification。

### 6.5 与 EVP stage1/2 的组合

PGC 与 EVP 位于不同位置：

```text
EVP stage1/2 -> 调整 backbone 中前两阶段视觉特征
PGC-LDP      -> 在 final mask/query 层融合 structure 与 CLIP semantic token
```

当前组合实验只额外启用视觉 EVP：

- `EVP_INPUT_TYPE: fft`；
- `EVP_PROMPT_TYPE: highpass`；
- `EVP_TUNING_STAGE: 12`；
- `EVP_FUSION_TYPE: gated_add`；
- `EVP_FREQUENCY_ROUTER: low_conditioned_band_amplitude_gate`；
- `EVP_FREQUENCY_MODULATION_STAGE: 12`；
- EVP text adapter 和 text geometry loss 关闭。

配置中的整数 `12` 是 stage 1 与 stage 2 的紧凑编码，不表示“第十二阶段”。

因此组合结果只能说明“PGC + 当前 EVP stage1/2 配置”的效果，不能当作纯 PGC 的结果。

## 7. 参数量与计算量

### 7.1 统计口径

以下按单图 `Q=250`、`D=640`、`h=8` 计算：

- `1 MAC = 2 FLOPs`；
- 包含 structure/CLIP projection、Q/K/V/O 和 attention matmul；
- 不包含 bias add、LayerNorm、softmax、mask resize/pooling、分类器、loss 和 backward；
- base 行表示“不增加 LDP core”，不是整个 FC-CLIP 模型的总计算量。

### 7.2 PGC 参数拆分

| 部件 | 参数量 |
| --- | ---: |
| `structure_projection: 256 -> 640` | 164,480 |
| `clip_projection: 640 -> 640` | 410,240 |
| `query_projection: 640 -> 640` | 410,240 |
| shared `key_projection: 640 -> 640` | 410,240 |
| `structure_value_projection: 640 -> 640` | 410,240 |
| `output_projection: 640 -> 640` | 410,240 |
| `LayerNorm(640)` | 1,280 |
| **合计** | **2,216,960** |

### 7.3 与 base 和原 global LDP 对比

| 方案 | 相对 base 新增参数 | FP32 参数大小 | LDP core MAC/图 | LDP core FLOPs/图 |
| --- | ---: | ---: | ---: | ---: |
| FC-CLIP base | 0 | 0 | 0 | 0 |
| 原 global LDP | 2,626,560 | 10.02 MiB | 0.735360G | 1.470720G |
| PGC-LDP | 2,216,960 | 8.46 MiB | 0.656000G | 1.312000G |

PGC 相对原 global LDP：

- 减少 409,600 个参数，即 `15.59%`；
- 减少 0.079360G MAC/图，即 `10.79%`；
- FP32 权重减少约 1.56 MiB。

attention score 数量为：

```text
global LDP: 8 x 250 x 250 = 500,000 / image
PGC-LDP:    8 x 250 x 2   =   4,000 / image
```

score tensor 缩小 125 倍、减少 99.2%，但 projection 是 core MAC 的主要来源，因此
不能将“score 数量缩小 125 倍”写成“整个模块计算量缩小 125 倍”。

训练阶段相对 base 还新增预测 mask 的 CLIP pooling；推理阶段 base 本来就需要该 pooling，
PGC 会复用其结果。端到端吞吐和显存必须通过 profiler 实测，不能由上述解析 MAC 直接替代。

## 8. 配置与启动

### 8.1 纯 PGC-LDP

现有 5k 配置为：

```text
configs/coco/panoptic-segmentation/fcclip/
  fcclip_convnext_base_waterovs_id_ldp_paired_gated_correction_noamp_3gpu_bs15_smoke_5k.yaml
```

其核心条件为：

```text
WaterOVS ID
seed=42
3 GPUs
global batch=15
FP32 / AMP=false
final-only PGC
hard detached pooling
preserve FC-CLIP void
EVP/text/image adapter disabled
```

下面的 50k 命令是基于当前配置层级整理的复现入口，不是历史 pure PGC 分段 resume
运行的逐字 provenance。历史 5k 的 branch/commit/config/launcher 快照位于其输出目录的
`实验记录.md` 和 `provenance/`；5k 之后的 resume 命令以 `log.txt` 为准。

从头训练总计 50k 可使用：

```bash
cd /data/gjf/code/fc-clip-amp

CUDA_VISIBLE_DEVICES=0,1,2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/data/gjf/env/miniconda3/envs/MARIS/bin/python train_net.py \
  --num-gpus 3 \
  --config-file \
  configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_base_waterovs_id_ldp_noamp_3gpu_bs15_full.yaml \
  MODEL.FC_CLIP.LDP.TYPE paired_gated_correction \
  SEED 42 \
  SOLVER.MAX_ITER 50000 \
  SOLVER.CHECKPOINT_PERIOD 5000 \
  TEST.EVAL_PERIOD 10000 \
  OUTPUT_DIR output/fcclip_convnext_base_waterovs_id_pgc_ldp_noamp_3gpu_bs15_full
```

### 8.2 PGC-LDP + EVP stage1/2 full

对应基础配置为：

```text
configs/coco/panoptic-segmentation/fcclip/
  fcclip_convnext_base_waterovs_id_ldp_preserve_void_evp_stage12_noamp_3gpu_bs15_full.yaml
```

独立且不存在的输出目录上的 `nohup` 启动方式为：

```bash
cd /data/gjf/code/fc-clip-amp

OUTPUT_DIR=output/fcclip_convnext_base_waterovs_id_ldp_preserve_void_evp_stage12_paired_gated_correction_noamp_3gpu_bs15_full_rerun
test ! -e "${OUTPUT_DIR}" || { echo "Output exists: ${OUTPUT_DIR}" >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}"

nohup env \
  CUDA_VISIBLE_DEVICES=0,1,2 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /data/gjf/env/miniconda3/envs/MARIS/bin/python train_net.py \
  --num-gpus 3 \
  --config-file \
  configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_base_waterovs_id_ldp_preserve_void_evp_stage12_noamp_3gpu_bs15_full.yaml \
  SOLVER.MAX_ITER 50000 \
  SOLVER.CHECKPOINT_PERIOD 5000 \
  TEST.EVAL_PERIOD 10000 \
  MODEL.FC_CLIP.LDP.TYPE paired_gated_correction \
  SEED 42 \
  OUTPUT_DIR "${OUTPUT_DIR}" \
  > "${OUTPUT_DIR}/nohup.log" 2>&1 &

echo $!
```

已有输出目录不得直接从头覆盖；需要续训时应先确认 `last_checkpoint` 和保存配置，再使用
`--resume`。

当前组合实验的实际 command line 已保存在其 `log.txt` 开头。该次运行以已有 stage1/2
saved config 为输入，并通过 CLI 覆盖 `LDP.TYPE` 和独立 `OUTPUT_DIR`；精确复现时应优先
读取该日志和输出目录的 `config.yaml`，而不是仅依赖上面的通用命令。

## 9. 代码验证

`test/test_local_discriminative_pooling.py` 已覆盖：

- PGC 输出 shape 和 finite 检查；
- attention weights shape `[B,Q,h,1,2]`；
- self value 逐元素严格为零；
- 将 V projection bias 人为设为非零后 self value 仍严格为零；
- structure/CLIP 两路输入均获得 finite、非零梯度；
- 所有 PGC 参数均获得 finite gradient；
- `Q=250,D=640,h=8` 的真实维度 FP32 forward/backward；
- PGC 参数量严格等于 2,216,960；
- builder 对 `global`、`paired_gated_correction` 和非法类型的处理；
- preserve-void logits 组合与梯度边界；
- decoder 只暴露 final `pred_region_features`。

实现完成时的验证记录为：

- LDP 定向测试 21 项通过；
- 仓库 `test/` 全套 138 项通过；
- 5k 启动和训练未出现 OOM、NaN、Inf 或 traceback。

## 10. WaterOVS ID 实测结果

### 10.1 对照

已有对照如下：

| 实验 | iterations | AP | AP50 | AP75 | APs | APm | APl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 历史 FC-CLIP base | 50k | 44.8520 | — | — | — | — | — |
| 原 global LDP preserve-void | 5k | 42.4391 | 51.8466 | 45.6741 | 0.1346 | 20.2589 | 43.8725 |
| 原 global LDP preserve-void | 50k | 49.0105 | 56.8701 | 52.5404 | 6.6625 | 28.6557 | 50.1719 |

历史 base 使用 AMP 和 global batch 16；LDP/PGC 使用 FP32 和 global batch 15。因此
base 只作为方向性参考，不能用于严格计算 PGC 的单变量增益。

### 10.2 纯 PGC-LDP

纯 PGC 的 `metrics.json` 实测为：

| iterations | AP | AP50 | AP75 | APs | APm | APl |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5k | 42.2373 | 50.8375 | 45.6565 | 0.0118 | 21.4407 | 43.6843 |
| 20k | 48.2816 | 56.4084 | 52.3338 | 1.4432 | 27.9573 | 49.5684 |
| 30k | 49.1416 | 57.3638 | 52.6436 | 6.4922 | 28.2935 | 50.5449 |
| **40k** | **49.4622** | **57.3764** | **53.2002** | **6.4924** | **28.4242** | **50.8465** |
| 50k final | 47.9295 | 56.1407 | 51.2522 | 0.0454 | 26.4964 | 49.0286 |

10k checkpoint 存在，但该次 resume 中没有对应 evaluation 记录，因此表中不补推测值。

实测事实：

- 5k PGC 比同设置 global LDP 低 0.2018 AP；
- 纯 PGC 最佳已测 checkpoint 是 40k，AP 为 49.4622；
- 50k final 比原 global LDP 50k 低 1.0810 AP；
- 50k 相比 PGC 40k 回落 1.5327 AP，并伴随 APs 从 6.4924 降至 0.0454。

因此不能仅以 PGC 40k best 宣称结构稳定优于 global LDP；当前证据说明 PGC 能达到
相近甚至略高的中期 AP，但 final checkpoint 存在明显回落，需要进一步诊断训练后期稳定性。

证据目录：

```text
output/fcclip_convnext_base_waterovs_id_ldp_paired_gated_correction_noamp_3gpu_bs15_smoke_5k/
  metrics.json
  log.txt
  model_0039999.pth
  model_final.pth
```

该目录最初用于 5k smoke，后来通过 `--resume` 继续到 50k。目录名中的 `smoke_5k`
不能代表其当前只包含 5k 结果，读取结果时必须使用 `metrics.json` 的 `iteration` 字段。

### 10.3 PGC-LDP + EVP stage1/2

截至本文撰写时已完成的 evaluation 为：

| iterations | AP | AP50 | AP75 | APs | APm | APl |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | 47.7752 | 56.3569 | 51.5917 | 3.2558 | 27.7101 | 49.5572 |
| 20k | 48.8433 | 57.0774 | 52.7319 | 3.2461 | 27.7981 | 50.5207 |
| **30k** | **50.5597** | **58.5395** | **54.3553** | **6.6479** | **31.3117** | **52.2128** |

30k 与纯 PGC 30k 的同迭代对比为：

| 指标 | 纯 PGC 30k | PGC + EVP stage1/2 30k | 差值 |
| --- | ---: | ---: | ---: |
| AP | 49.1416 | 50.5597 | +1.4181 |
| AP50 | 57.3638 | 58.5395 | +1.1757 |
| AP75 | 52.6436 | 54.3553 | +1.7117 |
| APs | 6.4922 | 6.6479 | +0.1557 |
| APm | 28.2935 | 31.3117 | +3.0182 |
| APl | 50.5449 | 52.2128 | +1.6679 |

这是当前最强的阶段性信号，尤其是 AP75、APm 和 APl；但仍有以下边界：

- 只有单 seed；
- pure PGC 经历过分段 resume，组合实验从头连续训练；
- 组合实验的 40k evaluation 与 50k final evaluation 在本文快照时尚未完成；
- periodic evaluation 的 `instance_inference` 在部分大样本上触发 CUDA OOM，
  Detectron2 `retry_if_cuda_oom` 将失败操作转到 CPU 后继续；10k–30k evaluation 均完成，
  但该 fallback 明显降低评估速度并暴露了推理显存压力；
- 该结果支持继续完成 full 和做消融，但不能提前写成最终 PGC 增益。

证据目录：

```text
output/fcclip_convnext_base_waterovs_id_ldp_preserve_void_evp_stage12_paired_gated_correction_noamp_3gpu_bs15_full/
  config.yaml
  metrics.json
  log.txt
  model_0029999.pth
  model_0039999.pth
```

## 11. 结果解释

### 11.1 已由实验证明的内容

- hard mask-wise pairing 能完成真实 WaterOVS ID 50k 训练且未发生数值崩溃；这不表示
  后期 AP 稳定；
- correction-only V、显式零 self value 和 preserve-void 没有导致训练启动失败或 AP 级别的
  catastrophic void collapse；当前没有独立的 void/objectness 统计；
- PGC 用更少的 LDP 参数和解析 MAC，在 30k/40k 可以达到原 global LDP 50k 附近的 AP；
- 当前 EVP stage1/2 与 PGC 组合在单 seed 的 30k 数值上比同迭代纯 PGC 高 1.4181 AP。

### 11.2 合理但尚未验证的解释

- paired topology 可能减少了跨 mask 的 structure 误写入；
- correction-only gate 可能比标准 MPCA 更好地保留 CLIP semantic anchor；
- EVP stage1/2 可能改善了水下图像的低层视觉适应，从而给 PGC 提供更好的 structure
  和 dense CLIP feature；
- PGC 50k 回落可能与训练后期过拟合、checkpoint 波动、小目标 query 退化或分段 resume
  有关。

这些解释目前没有 gate 分布、feature drift、matched-query 分类和多 seed 统计支持，不能写成
已证实机制。

## 12. 已知限制

- 两路输入最终同源于同一个 CLIP image encoder，不具备 SAM+CLIP 的异构互补性；
- paired topology 完全去掉 off-diagonal proposal context，可能损失有用的跨 query 信息；
- hard detached pooling 丢失 mask uncertainty，并切断 classification loss 到 mask 权重的直接梯度；
- residual 使用可训练 `clip_projection(G_clip)`，不是 raw CLIP anchor；
- PGC 当前只应用于 final decoder 层；
- `forward_with_attention_weights` 可提供 gate，但训练/评估流程尚未自动汇总 gate 统计；
- 解析 MAC 不包含 pooling、backbone、decoder、backward 和 DDP 通信；
- 当前 PGC+EVP stage1/2 的周期评估存在可恢复的 `instance_inference` CUDA OOM/CPU
  fallback；这不是训练 OOM，但需要单独优化评估显存；
- 当前结果只有 WaterOVS ID、ConvNeXt-B、seed 42，不能直接外推到其他数据集或 backbone；
- 纯 PGC 50k final 回落，说明仅报告 best checkpoint 会高估稳定性。

## 13. 后续验证优先级

建议按以下顺序继续，避免一次改变多个变量：

1. 完成 `PGC + EVP stage1/2` 的 40k evaluation 与 50k final evaluation，并同时报告
   best 与 final；
2. 对 40k 和 50k checkpoint 统计 `a^S` 的 per-head 均值、方差、熵和类别/尺寸分组；
3. 比较 PGC 40k 与 50k 的有效 object query、小目标 matched query 和 APs 退化来源；
4. 在相同 seed/训练拓扑下复跑 pure PGC，排除分段 resume 对 final 波动的影响；
5. 增加至少 3 seeds，报告均值与标准差；
6. 单独测试 raw-CLIP anchor residual：`LN(G_clip + correction)`；
7. 在 full-dim PGC 稳定后，再测试 internal dim 320/256 bottleneck；
8. 将 standard MPCA、PGC 和 global LDP 放在完全一致的训练/评估协议下对照。

## 14. 代码位置

| 文件 | 职责 |
| --- | --- |
| `fcclip/modeling/local_discriminative_pooling.py` | global LDP、PGC、输入校验、builder、preserve-void 组合 |
| `fcclip/modeling/transformer_decoder/fcclip_transformer_decoder.py` | final structure feature 和 hard detached `MaskPooling` |
| `fcclip/fcclip.py` | 构建 PGC、CLIP mask pooling、final logits 替换、推理 feature 复用 |
| `fcclip/config.py` | `LDP.TYPE`、head、dropout 默认配置 |
| `test/test_local_discriminative_pooling.py` | shape、zero self V、bias、gradient、参数量和 builder 测试 |
| `configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_base_waterovs_id_ldp_paired_gated_correction_noamp_3gpu_bs15_smoke_5k.yaml` | 纯 PGC 5k 配置 |
| `configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_base_waterovs_id_ldp_preserve_void_evp_stage12_noamp_3gpu_bs15_full.yaml` | PGC+EVP stage1/2 的基础配置，需显式覆盖 PGC type |

## 15. 当前结论

PGC-LDP 已经实现了预期的 keep-or-correct 语义：CLIP token 是 receiver 和 residual
anchor，self key 对应严格零 value，配对 structure token 是唯一 correction 来源。它相对
原 global LDP 减少 15.59% 模块参数和 10.79% core MAC，并将 attention score 从
500,000 降到 4,000/图。

实验上，纯 PGC 在 40k 达到 49.4622 AP，但 50k final 回落至 47.9295，尚不能证明其
稳定优于原 global LDP。当前更积极的证据来自 `PGC + EVP stage1/2`：30k 达到
50.5597 AP，比同迭代纯 PGC 高 1.4181 AP。该信号值得完成 full 并进一步诊断，但最终
结论必须等待 40k evaluation、50k final evaluation、多 seed 和 gate/feature 机制分析。

## Material Passport

- Origin: 当前仓库源码、配置、测试与本地训练产物
- Document Date: 2026-08-12
- Implementation Status: IMPLEMENTED
- Pure PGC Verification: 50k COMPLETED
- PGC + EVP stage1/2 Verification: RUNNING；本文快照包含至 30k 已完成 evaluation
- External Retrieval: none
