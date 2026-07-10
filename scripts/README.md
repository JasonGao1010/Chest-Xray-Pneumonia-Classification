# 脚本说明

本目录只保留复现和演示需要的主要脚本。

## 数据检查

```bash
python scripts/check_dataset.py --verify-images
```

该脚本检查 `data/raw/chest_xray/` 下的 `train`、`val`、`test` 目录、类别目录、图像数量和可读取性。

## 模型训练

```bash
python scripts/train.py \
  --model torchvision:vit_b_16 \
  --pretrained \
  --epochs 4 \
  --batch-size 16 \
  --learning-rate 0.00005
```

重新训练会受到硬件、依赖版本和随机性的影响。公开复现时，建议优先使用已经发布的 `models/vit_b16_best.pt` 进行测试集评价。

## 测试集评价

```bash
python scripts/evaluate.py \
  --checkpoint models/vit_b16_best.pt \
  --split test \
  --device cpu \
  --json-output results/eval_test_reproduced_vit_b16.json \
  --predictions-output results/predictions_test_reproduced_vit_b16.csv \
  --confusion-matrix-output figures/confusion_matrix_test_reproduced_vit_b16.png \
  --no-update-latest
```

该脚本会输出整体指标、逐图预测结果和混淆矩阵。

## 阈值分析

```bash
python scripts/analyze_thresholds.py \
  --predictions results/predictions_test_reproduced_vit_b16.csv \
  --json-output results/threshold_analysis_test_reproduced_vit_b16.json \
  --curve-output results/threshold_curve_test_reproduced_vit_b16.csv \
  --figure-output figures/threshold_curve_test_reproduced_vit_b16.png
```

该脚本根据测试集预测概率扫描分类阈值，用于观察 precision、recall、specificity 和 F1 的权衡。测试集阈值扫描只适合做结果分析，不应被解释为无偏部署阈值。

## RSNA 外部数据集

RSNA 原始数据不随仓库发布。下载 Kaggle 的 RSNA Pneumonia Detection Challenge 后，将文件放为：

```text
data/raw/rsna_pneumonia/
  stage_2_train_images/*.dcm
  stage_2_train_labels.csv
  stage_2_detailed_class_info.csv
```

转换为二分类数据：

```bash
python scripts/prepare_rsna_binary.py
python scripts/check_external_dataset.py --data-root data/processed/rsna_binary --verify-images
```

如果 Kaggle 竞赛数据需要认证，可使用 Hugging Face 镜像子集下载脚本。本轮扩容实验使用较大的候选池下载，随后只整理本地真实存在的 DICOM：

```bash
python scripts/download_rsna_hf_subset.py \
  --train-per-class 650 \
  --val-per-class 180 \
  --test-per-class 360
python scripts/prepare_rsna_binary.py --images-dir data/raw/rsna_pneumonia --available-only
python scripts/check_external_dataset.py \
  --data-root data/processed/rsna_binary \
  --json-output results/rsna_dataset_check.json \
  --verify-images
```

`prepare_rsna_binary.py` 会生成固定 seed=42 的 train / val / test 划分、`metadata.csv`、每个 split 的 `samples_*.csv` 和类别分布图。评价外部 test 时可复用主评价脚本：

```bash
python scripts/evaluate.py \
  --checkpoint models/vit_b16_best.pt \
  --data-root data/processed/rsna_binary \
  --split test \
  --device cpu \
  --json-output results/eval_rsna_test_vit_b16.json \
  --predictions-output results/predictions_rsna_test_vit_b16.csv \
  --confusion-matrix-output figures/confusion_matrix_rsna_test_vit_b16.png \
  --no-update-latest
```

如果 RSNA 下载受限，可按提升计划的第二优先路线使用 NIH ChestX-ray14 弱标签子集。当前脚本支持从 NIH metadata 和 `images_001.zip` 中构建 `Pneumonia` vs `No Finding` 二分类子集：

```bash
python scripts/prepare_nih_binary.py
python scripts/check_external_dataset.py \
  --data-root data/processed/nih_binary \
  --json-output results/nih_dataset_check.json \
  --verify-images
```

Kermany+RSNA 混合训练数据可由通用脚本生成：

```bash
python scripts/prepare_mixed_binary.py \
  --source-a data/raw/chest_xray \
  --source-b data/processed/rsna_binary \
  --prefix-a kermany \
  --prefix-b rsna \
  --output-root data/processed/kermany_rsna_binary \
  --summary-output results/kermany_rsna_dataset_summary.json \
  --splits train val
```

混合训练和 domain-balanced 采样可直接使用训练脚本。`domain-balanced-prefixes` 依赖混合数据文件名中的 `kermany_` 和 `rsna_` 前缀：

```bash
python scripts/train.py \
  --config configs/densenet121.yaml \
  --data-root data/processed/kermany_rsna_binary \
  --init-checkpoint outputs/checkpoints/20260709_225041_torchvision:densenet121/best.pt

python scripts/train.py \
  --config configs/densenet121.yaml \
  --data-root data/processed/kermany_rsna_binary \
  --domain-balanced-prefixes kermany rsna \
  --init-checkpoint outputs/checkpoints/20260709_225041_torchvision:densenet121/best.pt
```

冻结分类头和最后 block 解冻微调：

```bash
python scripts/train.py \
  --config configs/densenet121.yaml \
  --data-root data/processed/rsna_binary \
  --init-checkpoint outputs/checkpoints/20260709_225041_torchvision:densenet121/best.pt \
  --freeze-mode classifier

python scripts/train.py \
  --config configs/densenet121.yaml \
  --data-root data/processed/rsna_binary \
  --init-checkpoint outputs/checkpoints/20260709_225041_torchvision:densenet121/best.pt \
  --freeze-mode last_block
```

## 概率校准与错误样本

计算 ECE、Brier score 和可靠性曲线：

```bash
python scripts/analyze_calibration.py \
  --predictions results/predictions_test_vit_b16.csv \
  --json-output results/calibration_kermany_test_vit_b16.json \
  --bins-output results/calibration_bins_kermany_test_vit_b16.csv \
  --figure-output figures/reliability_kermany_test_vit_b16.png
```

如果已有独立校准集预测文件，可以用 `--calibration-predictions` 学习 temperature scaling。不要用同一份 test predictions 学温度参数。

导出错误样本、高置信错误和边界样本：

```bash
python scripts/analyze_errors.py \
  --predictions results/predictions_test_vit_b16.csv \
  --data-root data/raw/chest_xray \
  --csv-output results/error_cases_kermany_test_vit_b16.csv \
  --json-output results/error_summary_kermany_test_vit_b16.json \
  --grid-output figures/error_case_grid_kermany_test_vit_b16.png
```

汇总多实验结果：

```bash
python scripts/compare_experiments.py \
  --eval-json results/eval_test_reproduced_vit_b16.json \
  --calibration-json results/calibration_kermany_test_vit_b16.json
```

生成按真实类别分组的概率分布图：

```bash
python scripts/plot_probability_distribution.py \
  --predictions results/predictions_rsna_test_vit_b16.csv \
  --output figures/probability_distribution_rsna_test_vit_b16.png
```

## 网页演示

```bash
python scripts/serve_patient_app.py --checkpoint models/vit_b16_best.pt --device cpu
```

启动后打开：

```text
http://127.0.0.1:7860
```

网页演示只用于课程展示和研究训练，不用于临床诊断。
