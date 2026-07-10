# 脚本说明

所有命令从仓库根目录执行。正式数据和结果应优先使用只读核验或写入 `rebuild/`，避免覆盖当前证据。

## 自动测试

```bash
pytest -q
python -m pytest -q
```

两种入口均应得到相同结果。当前为 42 项通过。

## Kermany 数据与清单

检查官方目录图像：

```bash
python scripts/check_dataset.py --verify-images
```

核验现有保守文件名簇清单和 5856 个文件，不重建数据：

```bash
python scripts/prepare_kermany_grouped.py --verify-only --verify-files
```

数据准备脚本默认拒绝覆盖。实验性重建应使用新路径：

```bash
python scripts/prepare_kermany_grouped.py \
  --output-root rebuild/kermany_grouped_seed42 \
  --manifest rebuild/kermany_grouped_seed42.csv \
  --summary rebuild/kermany_grouped_summary.json
```

`--force` 会删除命令指定的旧输出，只能在明确确认路径可替换时使用。该方案按文件名派生簇隔离，不表示已验证患者独立。

## RSNA 数据与成员冻结

`prepare_rsna_binary.py` 把可用 DICOM 转为二分类 PNG，并按 `patientId` 分配集合。默认同样拒绝覆盖；`--dry-run` 不写任何文件。

```bash
python scripts/prepare_rsna_binary.py \
  --images-dir data/raw/rsna_pneumonia \
  --available-only \
  --output-root rebuild/rsna_binary \
  --splits-output rebuild/rsna_binary_seed42.json \
  --summary-output rebuild/rsna_dataset_summary.json \
  --figure-output rebuild/rsna_class_distribution.png
```

冻结当前 1707 个成员到独立清单：

```bash
python scripts/freeze_rsna_manifest.py \
  --manifest rebuild/rsna_available_1707_manifest.csv \
  --summary rebuild/rsna_available_1707_manifest_summary.json
```

脚本记录 `patientId`、标签、集合、来源记录、原始 DICOM 与处理后 PNG 的仓库相对路径、文件大小和 SHA-256。它不修改任何图像。

## 完整性审计

原始 Kermany 的保守裸前缀碰撞和 NIH 患者交叉按设计返回失败：

```bash
python scripts/audit_integrity.py \
  --dataset data/raw/chest_xray \
  --dataset data/processed/nih_binary \
  --output results/integrity_audit.json
```

修正分组 Kermany 与 RSNA 应通过：

```bash
python scripts/audit_integrity.py \
  --dataset data/processed/kermany_grouped_seed42 \
  --dataset data/processed/rsna_binary \
  --output results/integrity_audit_grouped.json
```

审计按数据集使用不同分组规则，并复算 `eval_*.json`、`strict_*`、`robust_*` 和 `mixed_*` 正式评价。原始 Kermany 的 170 个裸 `personN` 碰撞全部跨 `bacteria/virus` 子类型，不能解释为已确认患者泄漏。

## 训练

单个严格基线示例：

```bash
python scripts/train.py \
  --config configs/densenet121.yaml \
  --data-root data/processed/kermany_grouped_seed42 \
  --results-dir rebuild/results \
  --checkpoints-dir rebuild/checkpoints \
  --seed 42 \
  --json-output rebuild/results/strict_train_densenet121_seed42.json
```

稳健配方使用 `configs/densenet121_robust.yaml`。简单混合使用 `data/processed/kermany_grouped_rsna_mixed`；域均衡在同一命令上增加：

```text
--domain-balanced-prefixes kermany rsna
```

每个正式设置运行随机种子 42、43、44。训练脚本会在指定结果目录更新 `training_latest.json`，正式记录应使用显式 `--json-output` 文件。

## 评价

```bash
python scripts/evaluate.py \
  --training-summary rebuild/results/strict_train_densenet121_seed42.json \
  --data-root data/processed/kermany_grouped_seed42 \
  --split test --device auto \
  --json-output rebuild/results/strict_eval_kermany_test_densenet121_seed42.json \
  --predictions-output rebuild/results/strict_predictions_kermany_test_densenet121_seed42.csv \
  --confusion-matrix-output rebuild/figures/strict_confusion_kermany_test_densenet121_seed42.png \
  --no-update-latest
```

评价输出包括 JSON 指标、逐图预测 CSV 和混淆矩阵。主结果必须保留逐图预测，不能只保留手工表格。

## 三随机种子与成对统计

```bash
python scripts/summarize_strict_results.py \
  --bootstrap 5000 --bootstrap-seed 20260710
```

脚本对同一图像平均三个随机种子的阳性概率。Kermany 自助法使用 624 个保守的子类型不敏感文件名簇，与数据划分及完整性审计采用同一分组规则；RSNA 使用 442 个 `patientId`。候选方案和严格基线在同一次重采样中计算差值。区间不包含重新训练随机性。

## 来源可分性

```bash
python scripts/analyze_domain_shift.py
```

输出未控制标签、标签匹配、仅阴性和仅阳性四种来源分类结果。标签匹配 AUROC 是仓库保留的主诊断量；它只说明处理后来源可分，不证明设备或医院机制。

## 校准、阈值与错误样本

```bash
python scripts/analyze_calibration.py \
  --predictions results/strict_predictions_rsna_test_densenet121_seed42.csv \
  --calibration-predictions results/strict_predictions_rsna_val_densenet121_seed42.csv \
  --temperature-max 20 \
  --json-output rebuild/strict_calibration_rsna_seed42.json

python scripts/evaluate_frozen_thresholds.py \
  --validation results/strict_predictions_rsna_val_densenet121_seed42.csv \
  --test results/strict_predictions_rsna_test_densenet121_seed42.csv \
  --output rebuild/strict_operating_points_rsna_seed42.json

python scripts/analyze_errors.py \
  --predictions results/strict_predictions_rsna_test_densenet121_seed42.csv \
  --data-root data/processed/rsna_binary \
  --csv-output rebuild/strict_error_cases_rsna_seed42.csv \
  --json-output rebuild/strict_error_summary_rsna_seed42.json \
  --grid-output rebuild/strict_error_grid_rsna_seed42.png
```

温度只从验证预测学习。正温度不会改变二分类 0.5 阈值标签。错误网格采用极端 FP、极端 FN、阈值附近 FP 和阈值附近 FN 四桶固定配额，并在 JSON 中记录选择。

## 网页演示

```bash
python scripts/serve_patient_app.py \
  --checkpoint models/vit_b16_best.pt --device cpu --threshold 0.5
```

打开 `http://127.0.0.1:7860`。默认权重是旧 Kermany 官方划分 ViT-B/16 单模型，只用于研究演示，不代表严格主结果或临床能力。
