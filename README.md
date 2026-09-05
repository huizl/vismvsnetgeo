# Vis-MVSNetGeo

本目录用于研究 Vis-MVSNet 在大视差、遮挡和深度边界区域的改进。当前正式方法由三个独立模块组成，不再使用旧的 OA/Range 因子定义。

当前实验流程：`train.txt` 训练，训练期间与训练结束后均只在 `val.txt` 验证、选 checkpoint 和参数。后续实验不运行 `test.txt`。具体诊断与重训顺序见 [Val 实验流程](docs/VAL_WORKFLOW.md)。

## 三个模块

| 模块 | 名称 | 作用位置 |
|---|---|---|
| M1 | 深度假设感知源视图融合 | 为每个源视图、每个深度假设预测融合残差 |
| M2 | 几何监督的源视图可见性建模 | 训练逐源可见性头，并用保守软门控抑制遮挡源视图 |
| M3 | 覆盖保持的局部-扩展混合采样 | 保持中央深度假设间隔，只扩展少量尾部假设 |

统一融合权重为：

```text
w(s,d) = exp(-u(s) + r(s,d)) * ((1-beta) + beta*p_vis(s))
```

未启用某个模块时，对应项分别取 `r=0` 或可见性门控为 `1`。

M3 默认设置：

```text
Stage 2: 32 = 24 local + 8 wide
Stage 3: 16 = 12 local + 4 wide
```

低不确定性且未触及全局深度边界时，M3 与原始均匀级联假设完全一致；高不确定性时只移动尾部假设，中央局部假设保持原间隔。

## 八组消融

三位 code 固定按 `M1/M2/M3` 排列：

| `MODEL_TYPE` | code | M1 | M2 | M3 |
|---|---:|---:|---:|---:|
| `vis` | 000 | 0 | 0 | 0 |
| `m1_hyp` | 100 | 1 | 0 | 0 |
| `m2_visibility` | 010 | 0 | 1 | 0 |
| `m3_hybrid` | 001 | 0 | 0 | 1 |
| `m1_m2` | 110 | 1 | 1 | 0 |
| `m1_m3` | 101 | 1 | 0 | 1 |
| `m2_m3` | 011 | 0 | 1 | 1 |
| `full` | 111 | 1 | 1 | 1 |

唯一配置表位于 `models/model_variants.py`。训练、评测日志和区域 CSV 都使用相同 code。

## 目录

```text
models/model_variants.py             八组模块配置
models/vismvsnet.py                  原始 Vis-MVSNet 基线
models/vismvsnet_oa.py               M1/M2/M3 可配置模型与损失
models/dynamic_visibility.py         逐参考-源视图几何可见性标签
datasets/dtu_yao.py                  DTU 数据和可见性 GT 读取
train.py                             统一训练入口
tools/train_view5.sh                 单模型训练，支持 3 或 5 训练视图
tools/train_factorial_queue.sh       顺序训练多个模型
tools/eval_regions_view5.sh          单模型区域指标评测
tools/eval_region_metrics_dtu_yao.py 区域指标与 CSV 输出
tests/                               配置、损失、采样和完整前向测试
docs/METHOD.md                       方法公式、监督和实验顺序
```

## 单模型训练

五视图训练、五视图验证：

```bash
GPU=0 MODEL_TYPE=m2_visibility DATAPATH=/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu TRAINLIST=lists/dtu/train.txt VALLIST=lists/dtu/val.txt TRAIN_NVIEWS=5 EVAL_NVIEWS=5 LOGDIR=./checkpoints/dtu/m2_visibility_view5 BATCH_SIZE=4 EPOCHS=16 bash tools/train_view5.sh
```

三视图训练、五视图验证只需改变：

```bash
GPU=0 MODEL_TYPE=m2_visibility DATAPATH=/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu TRAINLIST=lists/dtu/train.txt VALLIST=lists/dtu/val.txt TRAIN_NVIEWS=3 EVAL_NVIEWS=5 LOGDIR=./checkpoints/dtu/m2_visibility_view3 BATCH_SIZE=4 EPOCHS=16 bash tools/train_view5.sh
```

主要默认参数：

```text
VISIBILITY_FUSION_BETA=0.2
HYPOTHESIS_RESIDUAL_SCALE=1.0
HYPOTHESIS_VISIBILITY_WEIGHT=0.1
HYBRID_STAGE2_WIDE_NUM=8
HYBRID_STAGE3_WIDE_NUM=4
HYBRID_SIGMA_SCALE=2.0
HYBRID_MAX_SCALE=2.0
HYBRID_CLIP_MODE=global
```

## 队列训练

默认顺序训练全部八组：

```bash
GPU=0 DATAPATH=/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu TRAINLIST=lists/dtu/train.txt VALLIST=lists/dtu/val.txt TRAIN_NVIEWS=5 EVAL_NVIEWS=5 LOG_ROOT=./checkpoints/dtu BATCH_SIZE=4 EPOCHS=16 bash tools/train_factorial_queue.sh
```

也可以只指定尚未训练的正式模型：

```bash
GPU=0 MODEL_TYPES='m2_visibility m3_hybrid m1_m2 m1_m3 m2_m3 full' DATAPATH=/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu TRAINLIST=lists/dtu/train.txt VALLIST=lists/dtu/val.txt TRAIN_NVIEWS=5 EVAL_NVIEWS=5 LOG_ROOT=./checkpoints/dtu BATCH_SIZE=4 EPOCHS=16 bash tools/train_factorial_queue.sh
```

每个目录保存：

```text
model_*.ckpt
latest.ckpt
best_abs.ckpt
best_2mm.ckpt
best_4mm.ckpt
best_8mm.ckpt
logs.txt
TensorBoard event files
```

## 区域评测

Val 示例：

```bash
GPU=0 MODEL_TYPE=m2_visibility CHECKPOINT=./checkpoints/dtu/m2_visibility_view5/best_2mm.ckpt LABEL=m2_visibility_view5 TESTLIST=lists/dtu/val.txt DATAPATH=/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu OUTDIR=./eval/ablation_val_light3/m2_visibility_view5 bash tools/eval_regions_view5.sh
```

`eval_regions_view5.sh` 默认 `TESTLIST=lists/dtu/val.txt`；变量名 `TESTLIST` 是历史接口，当前用于验证集。评测固定使用五个模型视图和五个区域定义视图，默认只测 `light=3`。

已有 checkpoint 的 Val 诊断（默认五视图训练模型，统一五视图推理）：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/validate_val_diagnostics.py --train_nviews 5 --suite all --outdir eval/val_diagnostics_view5_round1
```

脚本只读取 `lists/dtu/val.txt`，运行 vis 基线、M2 四组 beta 和 M3 四组裁剪/扩展对照。`--dry_run` 只展示命令；`--suite m2` 或 `--suite m3` 可只跑一个模块及基线。输出目录必须新建或为空，避免覆盖现有结果。

输出：

```text
all_metrics.csv      每张图、每个区域一行
summary_metrics.csv  像素加权平均与图像平均
```

区域包括：

```text
full
boundary
large_disparity
occluded_any
occluded_majority
large_disp_and_occluded
boundary_and_occluded
```

指标包括 `abs`、`acc2`、`acc4`、`acc8`，以及三级深度范围覆盖率和 Stage 2/3 范围宽度。

## Checkpoint 兼容

- 旧 `vis` checkpoint 可继续作为 `MODEL_TYPE=vis` 使用。
- 旧 `hyp` checkpoint 的参数结构与新 `m1_hyp` 相同，可用 `MODEL_TYPE=m1_hyp` 评测。
- `m2_visibility`、`m3_hybrid` 及其组合代表新方法语义，应重新训练。
- M3 没有新增可学习参数，但推理时必须使用正确的 `MODEL_TYPE` 和相同采样参数。
- M3 新增 `HYBRID_CLIP_MODE=global|none`；默认 `global` 复现旧结果，`none` 去掉全局裁剪。训练、评估均支持该选项，区域 CSV 记录 `hybrid_clip_mode`。同一 checkpoint 切换选项用于诊断，正式结果需保持训练与推理配置一致。

## 本地验证

```bash
python -m compileall -q models datasets train.py tools tests
python -m unittest discover -s tests -v
```

Windows 本地主要用于静态检查和 CPU 单测；完整训练与 DTU 区域评测在 CUDA 服务器运行。
