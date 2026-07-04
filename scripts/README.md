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

## 网页演示

```bash
python scripts/serve_patient_app.py --checkpoint models/vit_b16_best.pt --device cpu
```

启动后打开：

```text
http://127.0.0.1:7860
```

网页演示只用于课程展示和研究训练，不用于临床诊断。
