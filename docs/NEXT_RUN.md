# 下一次只运行基线范围失效诊断

目的：分清当前大误差与搜索范围排除 GT 的关系，以及范围内还剩多少匹配误差。使用既有 v2_vis checkpoint，不训练新模型、不改融合/采样，不进行 GT oracle 干预。

## 同步到服务器

这三个文件必须一起同步到服务器 vismvsnetgeo 的对应位置：

- `tools/eval_region_metrics_dtu_yao.py`：已有评估入口，新增可选 `--failure_diagnostics`。
- `tools/failure_diagnostics.py`：原始阶段分辨率的误差分组和报告。
- `tools/run_baseline_diagnostics.py`：固定基线、Val 列表和参数的一键入口。

其余模型和数据集文件沿用当前目录。服务器先激活原来的 MVS 训练环境。

## 运行命令

在服务器 `vismvsnetgeo` 目录执行：

```bash
# 先处理 8 张检查流程；此结果不能用于完整 Val 结论。
CUDA_VISIBLE_DEVICES=0 python tools/run_baseline_diagnostics.py --max_samples 8

# 上一步正常结束后，再运行完整 Val。
CUDA_VISIBLE_DEVICES=0 python tools/run_baseline_diagnostics.py
```

默认设置与已审计三视图基线的评估参数对应：

```text
checkpoint = checkpoints/v2/v2_vis_view3_seed1/best_2mm.ckpt
datapath   = /home/disk_10T/lzh_data/dtu_training/mvs_training/dtu
testlist   = lists/dtu/val.txt
model      = v2_vis
eval/region views = 5/5
light = 3; batch_size = 4; seed = 1
stage depth numbers = 48/32/16
stage interval scales = 4/2/1
```

`testlist` 是旧评估接口的参数名，此入口固定使用 val.txt。仅需调整路径时：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/run_baseline_diagnostics.py \
  --datapath /实际/DTU路径 \
  --checkpoint /实际/v2_vis_view3_seed1/best_2mm.ckpt
```

显存不足可加 `--batch_size 1`；实际命令会存进 manifest。`--dry_run` 只显示命令，不读取 DTU、不加载模型、不创建输出目录。每次默认创建独立的时间戳目录；指定 `--outdir` 时要求目录新建或为空。

## 结果文件

终端会打印 `eval/baseline_diagnostics/v2_vis_<时间戳>/` 的完整位置，其中包含：

| 文件 | 用途 |
| --- | --- |
| `failure_report.md` | 先阅读：各阶段、各区域范围内外的误差贡献 |
| `failure_summary.csv` | 像素加权分组指标、原始计数及误差和 |
| `failure_all.csv` | 逐 scan/view/region/stage/group 明细 |
| `summary_metrics.csv` / `all_metrics.csv` | 原评估指标，用于核对基线是否可比 |
| `diagnostic_manifest.json` | 完整命令、源码/列表哈希、样本限制、运行状态 |
| `diagnostic.log` | 完整运行日志 |

先确认 `diagnostic_manifest.json` 的 status=completed、scope=full_val，再把 `failure_report.md`、`failure_summary.csv` 和 `summary_metrics.csv` 一起用于分析。冒烟运行标为 smoke_only。

## 判断后续改动

- 若 S3 大视差/遮挡/交集中的范围外像素贡献较多误差，且到区间的距离下界也较大：优先诊断中心错误和候选丢失，再做单个候选保留改动。
- 若主要误差仍在范围内：下一步检查最近假设到 GT 的距离、双视图错误高权重、源视图遮挡。范围内不等于一定是融合错误，仍可能是采样、特征或正则化问题。
- 不仅按范围外像素比例做决定；少量远离搜索区间的像素可能贡献大量误差。

GT 和固定区域掩码用 nearest 映射到各阶段原始分辨率。各阶段样本格点不同，不把阶段间比例相减当作同一像素集的转移。区间误差下界不等于改大搜索范围后的实际收益；最近离散假设距离也不是连续 soft-argmin 的误差下界。

本地已完成新增统计/命令构造测试、既有范围指标测试、语法检查和 dry-run。当前本地 PyTorch 为 CPU 版，完整 DTU CUDA 运行需在服务器执行。
