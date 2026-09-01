# vismvsnetgeo

这是从旧 `vismvsnet` 实验目录中整理出的独立工程。旧目录不会被读取或修改；将本目录完整复制到服务器后即可训练和评测。

## 三个独立改进

- **A：逐源遮挡监督**。使用参考/源 GT 深度生成 source-specific 可见性标签，修正 pair depth、uncertainty 和原始 `occ_head` 的训练。
- **B：自适应深度范围**。使用上一阶段概率体标准差，在不确定像素扩大 Stage 2/3 搜索范围。
- **C：深度假设相关融合**。把原来对所有深度共享的源视图权重改成随深度假设变化的权重，并用真实逐源可见性监督融合头。

A、B、C 在代码中是独立开关，不是只能逐步叠加。

## 完整八组消融

| 顺序 | `MODEL_TYPE` | A | B | C | 含义 |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | `vis` | 0 | 0 | 0 | 原始 Vis-MVSNet 基线 |
| 2 | `oa` | 1 | 0 | 0 | 基线 + A |
| 3 | `range` | 0 | 1 | 0 | 基线 + B |
| 4 | `hyp` | 0 | 0 | 1 | 基线 + C |
| 5 | `oa_range` | 1 | 1 | 0 | 基线 + A + B |
| 6 | `oa_hyp` | 1 | 0 | 1 | 基线 + A + C |
| 7 | `range_hyp` | 0 | 1 | 1 | 基线 + B + C |
| 8 | `oa_full` | 1 | 1 | 1 | 基线 + A + B + C |

训练和评测启动时都会打印三位 `A/B/C` code；区域 CSV 也会保存三列因素开关，避免模型名称混淆。

## A：逐源遮挡监督

原 Vis-MVSNet 会在参考图有效区域监督每个 source pair，即使参考三维点在该源图中被遮挡。A 改为：

1. fused depth 仍监督所有参考有效像素；
2. pair-depth L1 只监督在当前源视图中真实可见的像素；
3. 遮挡像素的 pair error 只训练 uncertainty，该 error 会 detach，不向 pair depth 传播错误梯度；
4. 原始 `occ_head` 使用 source-specific 标签和 balanced focal BCE 训练。

关闭 A 时，统一损失中的 fused L1、pair L1 和 uncertainty NLL 与原始 `VisMVSLoss(occ_guide=False)` 数值一致。C 即使需要读取逐源 GT，也不会隐式打开 A。

## B：自适应级联范围

Stage 1 保持全局深度范围。Stage 2/3 根据上一阶段概率体标准差 `sigma` 生成逐像素范围：

```text
fixed_half = D_next / 2 * interval_next
half_width = clamp(k * sigma,
                   min_scale * fixed_half,
                   max_scale * fixed_half)
D_next(x) = linspace(depth_prev(x) - half_width(x),
                     depth_prev(x) + half_width(x))
```

默认 `k=2.0`、`min_scale=1.0`、`max_scale=2.0`。因此第一版只扩大高不确定区域，不会把搜索范围缩得比 baseline 更窄。它针对粗阶段错误在大视差、遮挡和边界区域造成的级联锁死。

B 不增加可学习参数，也不需要逐源可见性 GT。

## C：深度假设相关融合

原始融合权重 `exp(-uncertainty_i(x))` 在所有深度层共享。C 为每个 source pair 增加轻量 3D residual head：

```text
r_i(x,d) = scale * tanh(
    H([feature_i, score_i, probability_i, normalized_depth]))
log w_i(x,d) = -uncertainty_i(x) + r_i(x,d)
F(x,d) = sum_i w_i(x,d) * F_i(x,d) / sum_i w_i(x,d)
```

head 最后一层零初始化，所以初始化时 `r=0`，网络严格退化为原 uncertainty 融合。训练时，在离 GT 最近的深度假设处使用真实逐源可见性做 balanced focal BCE。该监督属于 C 本身；当 A=0 时，原始 pair loss 和 `occ_head` 仍保持 baseline 行为。

C 当前只支持项目默认的 `--vismode soft`。

## 目录

```text
vismvsnetgeo/
  datasets/                              DTU 数据和逐源 GT 深度/掩码
  lists/dtu/                             train、val、test 划分
  models/model_variants.py               八组 A/B/C 唯一配置表
  models/vismvsnet.py                    原始 baseline
  models/vismvsnet_oa.py                 A、B、C 模型与统一损失
  models/dynamic_visibility.py           逐源 GT 可见性投影
  tools/train_view5.sh                   单模型训练入口
  tools/train_factorial_queue.sh         单 GPU 顺序训练多组消融
  tools/eval_regions_view5.sh            单模型七区域评测入口
  tools/eval_region_metrics_dtu_yao.py   逐图和汇总 CSV
  tests/                                 损失、范围、融合和前向测试
  docs/METHOD.md                         论文方法与实验判据
```

Conv/PID/旧 Geo 组合、历史 checkpoint、旧输出与历史文档没有迁入。

## 数据职责

- `lists/dtu/train.txt`：训练；
- `lists/dtu/val.txt`：每轮验证并选择 `best_*.ckpt`；
- `lists/dtu/test.txt`：最终报告，只在模型与超参数确定后运行。

不要用 `test.txt` 选择 checkpoint。

## 先运行本地测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖八组配置、baseline 损失等价性、A 的遮挡梯度屏蔽、B 的范围扩展、C 的 GT 深度平面监督、checkpoint 参数兼容性，以及 A+B+C 的完整三级前向。

## 当前四卡安排

假设正在运行的 `vis` 和 `oa` 分别占用 GPU 0、1，则 GPU 2、3 各顺序运行三组剩余实验。开两个终端：

终端一：

```bash
GPU=2 MODEL_TYPES="range oa_range range_hyp" DATAPATH=/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu TRAINLIST=lists/dtu/train.txt VALLIST=lists/dtu/val.txt LOG_ROOT=./checkpoints/dtu BATCH_SIZE=4 EPOCHS=16 bash tools/train_factorial_queue.sh
```

终端二：

```bash
GPU=3 MODEL_TYPES="hyp oa_hyp oa_full" DATAPATH=/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu TRAINLIST=lists/dtu/train.txt VALLIST=lists/dtu/val.txt LOG_ROOT=./checkpoints/dtu BATCH_SIZE=4 EPOCHS=16 bash tools/train_factorial_queue.sh
```

每张 GPU 上的三组会依次运行，不会同时抢同一张卡。已有 `latest.ckpt` 的完整实验默认跳过。若实际空闲卡号不同，只改 `GPU`。

### 单独训练一组

```bash
GPU=2 MODEL_TYPE=range DATAPATH=/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu TRAINLIST=lists/dtu/train.txt VALLIST=lists/dtu/val.txt LOGDIR=./checkpoints/dtu/range_view5 BATCH_SIZE=4 EPOCHS=16 bash tools/train_view5.sh
```

多卡 DataParallel 使用英文逗号，例如 `GPU=2,3`。中文逗号不能被 CUDA 解析。`BATCH_SIZE` 是该进程的总 batch size，不是每张卡各自的 batch size。完整消融应保持相同的总 batch size、epoch、学习率和数据列表。

## 初始化与 checkpoint

论文主消融建议八组都从头训练。快速功能验证可以设置：

```bash
INIT_CKPT=./checkpoints/dtu/oa_view5/best_2mm.ckpt
```

- 不含 C 的模型 `vis/oa/range/oa_range` 参数结构相同；
- 含 C 的模型 `hyp/oa_hyp/range_hyp/oa_full` 参数结构相同，并只额外增加 hypothesis heads；
- 含 C 的模型可以加载不含 C 的 checkpoint，新增 head 保持零初始化；
- 含 C 的 checkpoint 必须以含 C 的 `MODEL_TYPE` 加载；
- B 没有参数，但推理时必须使用正确的 `MODEL_TYPE` 才会启用自适应范围；
- 从旧模型继续训练属于 fine-tuning，不应和“八组都从头训练”的主消融混在一起。

## 训练输出

每个 `LOGDIR` 保存：

- 每轮 `model_*.ckpt`；
- `best_abs.ckpt`；
- `best_2mm.ckpt`、`best_4mm.ckpt`、`best_8mm.ckpt`；
- `latest.ckpt`；
- `logs.txt` 和 TensorBoard 标量。

A 输出逐源可见/遮挡统计；B 输出每阶段范围覆盖率与范围宽度；C 输出 hypothesis visibility loss、accuracy、visible recall 和 occluded recall。

## 七区域评测

```bash
GPU=0 MODEL_TYPE=oa_full CHECKPOINT=./checkpoints/dtu/oa_full_view5/best_2mm.ckpt LABEL=oa_full_view5 TESTLIST=lists/dtu/test.txt OUTDIR=./eval/oa_full_test_light3 bash tools/eval_regions_view5.sh
```

输出：

```text
OUTDIR/
  all_metrics.csv
  summary_metrics.csv
```

两个 CSV 都保留 `full`、`boundary`、`large_disparity`、`occluded_any`、`occluded_majority`、`large_disp_and_occluded`、`boundary_and_occluded` 七个区域。指标包括像素数、Abs Error、Acc2/Acc4/Acc8、三个阶段的 GT-in-range 覆盖率，以及归一化搜索范围宽度。

先使用 `summary_metrics.csv` 完成八组总体消融，再从 `all_metrics.csv` 中挑选有代表性的 scan/view 做可视化。
