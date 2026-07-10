# 肺炎 X 光图像诊断识别

本仓库是《模式识别与机器学习》课程设计项目，任务是对胸部 X 光图像进行正常/肺炎二分类识别。仓库提供训练和评价代码、网页演示、主要结果文件和课程报告。

本项目只用于课程设计和研究训练，不构成临床诊断依据。

## 2026-07-10 严格审计与重建结果

最终审计发现，Kermany 官方目录中有 170 个肺炎文件名患者编号同时出现在 train 和 test。旧指标可以复算，但只能作为公开目录复现，不能视为严格患者独立评价。项目已按患者重新划分全部 5856 张图像：train 4107、validation 579、locked test 1170；新划分没有患者交叉、跨集合精确重复或标签冲突。

三种模型各运行 3 个随机种子。患者级内部测试的 accuracy 均值为 DenseNet121 0.9752、ConvNeXt-Tiny 0.9764、ViT-B/16 0.9781；直接迁移到 RSNA 后分别降到 0.6614、0.6840 和 0.6599，说明数据来源变化比结构差异更重要。

当前综合表现最好的提升方案是患者级 Kermany 与 RSNA 简单混合训练：

| 测试集 | Accuracy | Balanced accuracy | Recall | Specificity | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Kermany patient-level locked test | 0.9681±0.0044 | 0.9684±0.0040 | 0.9677±0.0053 | 0.9692±0.0049 | 0.9943±0.0012 |
| RSNA test | 0.7632±0.0107 | 0.7645±0.0095 | 0.7151±0.0592 | 0.8140±0.0426 | 0.8565±0.0026 |

这是有监督多域训练结果，不是对未知医院的零样本域泛化。RSNA 使用镜像可用子集，NIH 仍是小规模弱标签敏感性分析。

完整审查见 `reports/audit_review.md`，机器审计见 `results/integrity_audit.json`，多随机种子汇总见 `results/strict_summary.json`、`results/robust_summary.json` 和 `results/mixed_strict_summary.json`。

## 项目内容

公开发布包中建议保留以下内容：

```text
configs/                 主模型配置
data/README.md           数据集放置说明
figures/                 报告中使用的主要图表
models/README.md         模型权重放置说明
reports/                 课程报告及源码
results/                 主模型预测结果和指标摘要
sample_images/           网页演示用样例图
scripts/                 数据检查、训练、评价和网页演示脚本
src/                     可复用源码
tests/                   基础测试
web/                     本地网页演示
```

完整数据集和模型权重不随普通 Git 仓库提交。若要运行网页演示或复现主结果，需要先从 [v1.0.0 Release](https://github.com/JasonGao1010/Chest-Xray-Pneumonia-Classification/releases/tag/v1.0.0) 附件下载 `vit_b16_best.pt`，并放置为 `models/vit_b16_best.pt`。若要重新计算测试集指标，还需要自行准备同一数据集。

权重文件的 SHA256 校验值为：

```text
78604698f504d8d96f5bdad9570608c77472e7625684cb13a6f222275c8381be
```

## 环境准备

建议使用 Python 3.10 或更新版本。CPU 可以运行网页演示和评价流程；重新训练建议使用 CUDA GPU。

安装依赖：

```bash
pip install -r requirements.txt
```

如果使用独立虚拟环境，可以先创建环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell 下激活命令通常为：

```powershell
.venv\Scripts\Activate.ps1
```

## 数据集准备

本项目使用 Chest X-Ray Images (Pneumonia) 数据集。该数据集来自 Kermany 等发布的儿童胸部 X 光图像数据，原始发布页为 Mendeley Data `10.17632/rscbjbr9sj.2`，常用下载入口为 Kaggle 的 `paultimothymooney/chest-xray-pneumonia`。请将数据集整理为下面的目录结构：

```text
data/raw/chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

本报告使用的本地可读图像数量为 5856 张，其中训练集 5216 张、验证集 16 张、测试集 624 张。测试集包含 NORMAL 234 张、PNEUMONIA 390 张。

检查数据集是否放置正确：

```bash
python scripts/check_dataset.py --verify-images
```

如果数据集版本、目录划分或损坏文件处理不同，重新计算出的指标可能与报告存在差异。

## 运行网页演示

网页演示使用已经训练好的 ViT-B/16 主模型权重。请先确认权重文件位于：

```text
models/vit_b16_best.pt
```

然后运行：

```bash
python scripts/serve_patient_app.py --checkpoint models/vit_b16_best.pt --device cpu
```

启动后在浏览器打开：

```text
http://127.0.0.1:7860
```

可以使用 `sample_images/` 中的两张图片测试上传和推理流程。网页会输出 NORMAL 与 PNEUMONIA 两类概率，并按默认阈值 0.5 给出分类结果。

## 复现主结果

复现报告中的主结果需要先按上文准备完整数据集，然后运行评价脚本。

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

重新生成阈值分析：

```bash
python scripts/analyze_thresholds.py \
  --predictions results/predictions_test_reproduced_vit_b16.csv \
  --json-output results/threshold_analysis_test_reproduced_vit_b16.json \
  --curve-output results/threshold_curve_test_reproduced_vit_b16.csv \
  --figure-output figures/threshold_curve_test_reproduced_vit_b16.png
```

报告采用的主模型为 ImageNet 预训练 ViT-B/16，默认阈值为 0.5。测试集主指标如下：

| 指标 | 数值 |
| --- | ---: |
| 准确率 Accuracy | 0.9006 |
| 精确率 Precision | 0.8779 |
| 召回率 Recall | 0.9769 |
| 特异度 Specificity | 0.7735 |
| F1 | 0.9248 |
| ROC-AUC | 0.9710 |

对应混淆矩阵为：

| 真实类别 / 预测类别 | NORMAL | PNEUMONIA |
| --- | ---: | ---: |
| NORMAL | 181 | 53 |
| PNEUMONIA | 9 | 381 |

如果复现成功，`results/eval_test_reproduced_vit_b16.json` 中的指标应与上表基本一致。

## 提升版实验

本轮提升把当前 Kermany-only baseline 冻结在 `results/baseline_manifest.json`，新增结果均写入独立文件，不覆盖 `results/main_test_metrics.json`。

新增模型配置：

```text
configs/densenet121.yaml
configs/convnext_tiny.yaml
```

已补齐 ImageNet 预训练 DenseNet121 和 ConvNeXt-Tiny 的 Kermany test 评价，并把 RSNA Pneumonia Detection Challenge 的 Hugging Face 镜像子集扩容为外部测试集。NIH ChestX-ray14 `Pneumonia` vs `No Finding` 弱标签子集保留为第二个压力测试。

| 模型 | 预训练 | Acc | Prec | Rec | Spec | F1 | AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-B/16 | ImageNet | 0.9006 | 0.8779 | 0.9769 | 0.7735 | 0.9248 | 0.9710 |
| DenseNet121 | ImageNet | 0.9054 | 0.8770 | 0.9872 | 0.7692 | 0.9288 | 0.9743 |
| ConvNeXt-Tiny | ImageNet | 0.8878 | 0.8524 | 0.9923 | 0.7137 | 0.9171 | 0.9781 |

汇总表位于：

```text
results/experiment_comparison.csv
results/calibration_comparison.csv
```

概率可靠性分析已经补充 ECE、Brier score 和可靠性曲线。Kermany test 上，ViT-B/16 未校准 ECE 为 0.0645，Brier score 为 0.0850；DenseNet121 的 ECE 为 0.0548，Brier score 为 0.0752；ConvNeXt-Tiny 的 ECE 为 0.0776，Brier score 为 0.0897。

错误样本分析已经导出 false positive、false negative、高置信错误和边界样本。ViT-B/16 在默认阈值下共有 53 个 false positive、9 个 false negative，其中高置信 false positive 为 33 个，高置信 false negative 为 2 个。

```text
results/error_cases_kermany_test_vit_b16.csv
results/error_summary_kermany_test_vit_b16.json
figures/error_case_grid_kermany_test_vit_b16.png
figures/reliability_kermany_test_vit_b16_uncalibrated.png
figures/gradcam_kermany_error_cases_densenet121.png
figures/gradcam_rsna_positive_cases_densenet121.png
figures/gradcam_nih_positive_cases_densenet121.png
```

RSNA 外部子集来自 `Baldezo313/rsna-pneumonia-dataset` 镜像，按 seed=42 固定划分并转为二分类 PNG。当前扩容版共 1707 张，train 为 NORMAL 560 / PNEUMONIA 492，val 为 107 / 106，test 为 215 / 227。Kermany 训练模型直接迁移到扩容 RSNA test 后，DenseNet121 相对最好，但 accuracy 仍只有 0.7330，说明跨来源下降明显：

| 模型 | Acc | Prec | Rec | Spec | F1 | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-B/16 | 0.6923 | 0.6502 | 0.8678 | 0.5070 | 0.7434 | 0.7735 |
| DenseNet121 | 0.7330 | 0.6799 | 0.9075 | 0.5488 | 0.7774 | 0.8090 |
| ConvNeXt-Tiny | 0.7014 | 0.6518 | 0.8987 | 0.4930 | 0.7556 | 0.7865 |

扩容后的 DenseNet121 训练策略对照显示，不同策略的收益不是单向的。简单混合训练提高 RSNA accuracy 和 specificity，但降低 recall，并让 Kermany test accuracy 降到 0.8606；domain-balanced sampling 更好地保住 Kermany test，但 RSNA F1 低于直接迁移；冻结分类头和最后 block 解冻更偏向高 recall，仍没有解决正常样本误报问题。全模型低学习率微调也已补齐：RSNA test accuracy 为 0.7217、F1 为 0.7743、AUC 为 0.8343，回测 Kermany test 时 accuracy 为 0.8766、F1 为 0.9087。它没有形成双数据集共同提升，因此作为负结果保留。

| 训练数据 | 测试数据 | 模型 | Acc | Rec | Spec | F1 | AUC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| RSNA train | RSNA test | DenseNet121 | 0.7285 | 0.7533 | 0.7023 | 0.7403 | 0.8218 |
| Kermany+RSNA simple | RSNA test | DenseNet121 | 0.7602 | 0.6432 | 0.8837 | 0.7337 | 0.8568 |
| Kermany+RSNA domain-balanced | RSNA test | DenseNet121 | 0.7466 | 0.6035 | 0.8977 | 0.7098 | 0.8623 |
| RSNA frozen head | RSNA test | DenseNet121 | 0.7466 | 0.8811 | 0.6047 | 0.7812 | 0.8012 |
| RSNA last block | RSNA test | DenseNet121 | 0.7398 | 0.9119 | 0.5581 | 0.7826 | 0.8258 |
| RSNA full fine-tune | RSNA test | DenseNet121 | 0.7217 | 0.9295 | 0.5023 | 0.7743 | 0.8343 |
| Kermany+RSNA domain-balanced | Kermany test | DenseNet121 | 0.9022 | 0.9872 | 0.7607 | 0.9266 | 0.9742 |

NIH 弱标签子集来自 `images_001.zip`，包含 130 张图像，train/val/test 为 92/12/26，test 中 NORMAL 和 PNEUMONIA 各 13 张。Kermany 训练模型直接迁移到 NIH test 后表现也明显下降：

| 模型 | Acc | Prec | Rec | Spec | F1 | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-B/16 | 0.5000 | 0.5000 | 0.4615 | 0.5385 | 0.4800 | 0.5325 |
| DenseNet121 | 0.5385 | 0.5385 | 0.5385 | 0.5385 | 0.5385 | 0.6331 |
| ConvNeXt-Tiny | 0.4615 | 0.4667 | 0.5385 | 0.3846 | 0.5000 | 0.5621 |

扩容 RSNA test 已达到每类至少 200 张的最低外部验证门槛，但它仍来自镜像子集，且标签定义、年龄结构和图像格式都不同于 Kermany。NIH 标签又来自报告文本挖掘弱标签。因此这些结果适合说明跨来源泛化风险，不能写成临床诊断能力验证。

## 重新训练

如需从头训练，可使用：

```bash
python scripts/train.py \
  --model torchvision:vit_b_16 \
  --pretrained \
  --epochs 4 \
  --batch-size 16 \
  --learning-rate 0.00005
```

训练结果会受硬件、PyTorch/CUDA 版本和随机性影响。公开复现时，建议以 `models/vit_b16_best.pt` 对测试集重新评价作为主复现路径。

## 报告和结果文件

最终课程报告位于：

```text
reports/course_report.pdf
reports/thesis/main.tex
reports/thesis/sections/
```

最终 PDF 共 70 页，其中第1--12章连续正文52页；正文以数据审计、方法、实验、统计和讨论为主，不依靠单图附录凑页。

主结果摘要位于：

```text
results/main_test_metrics.json
results/predictions_test_vit_b16.csv
results/threshold_curve_test_vit_b16.csv
```

主要图表位于：

```text
figures/confusion_matrix_test_vit_b16.png
figures/threshold_curve_test_vit_b16.png
figures/probability_distribution_test_vit_b16.png
figures/error_profile_default_threshold.png
```

## 发布说明

主模型权重约 1GB，不建议直接提交到普通 Git 仓库。推荐做法是：代码、报告、主图和主结果文件进入仓库；`models/vit_b16_best.pt` 放入 GitHub Release 或使用 Git LFS 管理；完整数据集由复现者自行下载。
