# 肺炎 X 光图像诊断识别

本仓库是《模式识别与机器学习》课程设计项目，任务是对胸部 X 光图像进行正常/肺炎二分类识别。仓库提供训练和评价代码、网页演示、主要结果文件和课程报告。

本项目只用于课程设计和研究训练，不构成临床诊断依据。

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

完整数据集和模型权重不随普通 Git 仓库提交。若要运行网页演示或复现主结果，需要先从 Release 附件或课程交付包中取得 `vit_b16_best.pt`，并放置为 `models/vit_b16_best.pt`。若要重新计算测试集指标，还需要自行准备同一数据集。

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

本项目使用 Chest X-Ray Images (Pneumonia) 数据集。请将数据集整理为下面的目录结构：

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

课程报告位于：

```text
reports/course_report.pdf
reports/course_report.md
reports/course_report.tex
```

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
