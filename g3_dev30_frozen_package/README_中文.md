# G3 DEV30 冻结结果与统一 240×240 评估

本包包含两部分：

1. 已完成的 G3-Ensemble-CBSGuard DEV30 冻结结果；
2. 将 Original、Local-AA、Full-CBS、E3.5、G3-CBSGuard 统一到 240×240 的评价脚本。

当前版本已适配 OOF condition cache 的实际键名：`condition_speed` 与 `target_speed`。归一化数组 `condition_norm/target_norm` 不参与物理单位图像评价。

G3 候选读取 `ensemble_predictions/*` 中的 `hybrid_speed`。脚本还会与 `guarded_predictions/*` 中的 `raw_g3_speed_image`、`selected_speed_image` 和 `selected_g3` 做逐样本一致性校验。

240 网格评价严格复刻旧物理流水线：condition image-coordinate 转置为 physics-coordinate，再执行 `256→480→clip→240`；Original 与 GT 的权威 480 网格直接读取 `local_alpha01` 的 alpha=0/1 `target_480`。这一步不能替换为直接 `256→240`。

## 已冻结结果

- `dev30_evaluation.json`
- `dev30_method_summary.csv`
- `dev30_per_sample.csv`
- `paired_statistics.json`

这些文件对应 test21–50 的一次性冻结验证。不要再根据 DEV30 修改 flow/DDPM/CBS-guard 参数。

## 安装脚本

将下面三个文件复制到服务器 `USCT_download` 仓库根目录：

- `evaluate_dev30_all_methods_240.py`
- `inspect_g3_prediction_keys.py`
- `run_dev30_all_methods_240.sh`

然后执行：

```bash
chmod +x run_dev30_all_methods_240.sh
bash run_dev30_all_methods_240.sh 2>&1 | tee dev30_all_methods_240.log
```

正常会输出：

```text
Harmonized DEV30 comparison: 240x240
...
alignment max abs MSE error = ...
saved to: .../all_methods_harmonized_240
```

输出目录包含：

- `dev30_all_methods_240_per_sample.csv`
- `dev30_all_methods_240_summary.csv`
- `dev30_all_methods_240_statistics.json`
- `harmonization_audit.json`
- `dev30_all_methods_240_tradeoff.png`
- `g3_guarded_per_sample_gains_240.png`

## 如果提示找不到 G3 prediction key

先运行：

```bash
python inspect_g3_prediction_keys.py \
  --root ./dev30_test21_50/results/g3_ensemble_cbs_guard_frozen \
  --sample_id 21
```

把完整输出发回即可。评价脚本会在无法唯一确定数组时主动停止，不会猜测并生成错误结果。

## 实验纪律

- G3 的 true-CBS guard 选择直接复用已冻结的 `dev30_per_sample.csv`，不重新选择参数。
- G3 运行时间不写入统一表，因为当前第二次运行复用了 CBS 缓存。
- 统一脚本会用 Original MSE240 做逐样本对齐检查；对齐失败时禁止生成合并主表。
- 旧 240 物理流水线与 G3 流水线的 Original CBS-MSE 存在轻微差异，因此各方法的物理改善按其各自冻结运行中的 Original 基线计算；不得直接按不同流水线的绝对 physics-MSE 排名。
