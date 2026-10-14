# WaterOVS 注册与训练配置说明

## 文档定位

本文描述当前 `/data/gjf/output/fc-clip` 中 WaterOVS instance segmentation
注册和 FC-CLIP 配置。当前实现不移动、不复制 WaterOVS 原始数据，只引用 MARIS
仓库下的数据和类别定义。

## 数据集注册

注册文件：

```text
fcclip/data/datasets/register_waterovs_instance.py
```

注册名：

```text
openvocab_waterovs_train
openvocab_waterovs_val
```

默认 MARIS 仓库：

```text
/data/gjf/code/MARIS
```

默认 WaterOVS 数据根目录：

```text
/data/gjf/code/MARIS/datasets/WaterOVS
```

实际引用：

```text
/data/gjf/code/MARIS/datasets/WaterOVS/train
/data/gjf/code/MARIS/datasets/WaterOVS/val
/data/gjf/code/MARIS/datasets/WaterOVS/annotations/instances_train.json
/data/gjf/code/MARIS/datasets/WaterOVS/annotations/instances_val.json
```

类别定义来自：

```text
/data/gjf/code/MARIS/maris/data/datasets/openseg_classes.py
```

可用环境变量覆盖路径：

```bash
export MARIS_REPO_ROOT=/path/to/MARIS
export WATEROVS_DATASET_ROOT=/path/to/WaterOVS
```

如果 MARIS 类别定义文件不存在，注册阶段会抛出 `FileNotFoundError`，不会静默注册
空类别。

## 类别检查

当前注册阶段显式检查类别数量：

| dataset | 期望类别数 |
| --- | ---: |
| `openvocab_waterovs_train` | 84 |
| `openvocab_waterovs_val` | 115 |

同时会检查类别 id 去重，并构建 Detectron2 metadata：

```text
thing_dataset_id_to_contiguous_id
thing_classes
thing_colors
```

`thing_dataset_id_to_contiguous_id` 中的 dataset id 被格式化为三位字符串，以匹配
当前 MARIS 类别定义的 id 表达。

## Base 配置

Large backbone WaterOVS：

```text
configs/coco/panoptic-segmentation/fcclip/
  fcclip_convnext_large_waterovs_id.yaml
```

关键设置：

```yaml
MODEL:
  SEM_SEG_HEAD:
    NUM_CLASSES: 84
  MASK_FORMER:
    TEST:
      SEMANTIC_ON: false
      INSTANCE_ON: true
      PANOPTIC_ON: false

INPUT:
  DATASET_MAPPER_NAME: "coco_instance_lsj"

DATASETS:
  TRAIN: ("openvocab_waterovs_train",)
  TEST: ("openvocab_waterovs_val",)
```

Base backbone WaterOVS：

```text
configs/coco/panoptic-segmentation/fcclip/
  fcclip_convnext_base_waterovs_id.yaml
```

它继承 large WaterOVS 配置，并覆盖：

```yaml
MODEL:
  FC_CLIP:
    CLIP_MODEL_NAME: "convnext_base_w_320"
    EMBED_DIM: 640
```

训练 split 使用 84 个 seen 类别；验证 split 使用 115 个类别 metadata。FC-CLIP
推理时会根据测试集 metadata 重建文本分类器。

## LDP / PGC 配置关系

纯 LDP 50k 配置：

```text
fcclip_convnext_base_waterovs_id_ldp_noamp_3gpu_bs15_full.yaml
```

关键差异：

```yaml
MODEL:
  WEIGHTS: ""
  FC_CLIP:
    LDP:
      ENABLED: true
      NUM_HEADS: 8
      DROPOUT: 0.0

SOLVER:
  AMP:
    ENABLED: false
  IMS_PER_BATCH: 15
  MAX_ITER: 50000
```

PGC smoke 配置只覆盖 LDP 类型：

```yaml
MODEL:
  FC_CLIP:
    LDP:
      TYPE: "paired_gated_correction"
```

`*_preserve_void_*` 文件名用于保留早期实验命名；当前代码中 preserve-void 是固定
行为。

## EVP 配置关系

纯 EVP low-conditioned gated-add smoke 配置：

```text
fcclip_convnext_base_waterovs_id_evp_high_low_router_stage12_gated_add_gb2_smoke_5k.yaml
```

该配置继承 base WaterOVS，不启用 LDP/PGC。核心变量是：

```yaml
MODEL:
  FC_CLIP:
    EVP_ENABLED: true
    EVP_INPUT_TYPE: "fft"
    EVP_PROMPT_TYPE: "highpass"
    EVP_TUNING_STAGE: 12
    EVP_FUSION_TYPE: "gated_add"
    EVP_FREQUENCY_ROUTER: "low_conditioned_gate"
```

## 启动命令模板

在当前仓库运行：

```bash
cd /data/gjf/output/fc-clip
CUDA_VISIBLE_DEVICES=0,1,2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/data/gjf/env/miniconda3/envs/MARIS/bin/python train_net.py \
  --num-gpus 3 \
  --config-file configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_base_waterovs_id_ldp_noamp_3gpu_bs15_smoke_5k.yaml
```

如果要改输出目录，追加：

```bash
OUTPUT_DIR output/my_run_name
```

不要复用已有输出目录从头覆盖。需要续训时先检查该目录的 `last_checkpoint`、
`config.yaml` 和 `log.txt`，再使用 `--resume`。

## 证据边界

当前迁移验证只跑了短程训练日志，用于确认配置、数据注册、模型构建和 loss 计算正常。
完整实验必须在输出目录中记录：

```text
实验记录.md
log.txt
metrics.json
config.yaml
last_checkpoint
```

没有 evaluator-backed `metrics.json` 时，不要把配置名、loss 曲线或 checkpoint
存在本身转换成 AP 结论。
