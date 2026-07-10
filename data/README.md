# 数据集目录说明

完整数据集不随仓库发布。请自行下载 Chest X-Ray Images (Pneumonia) 数据集，并整理为以下目录结构：

## 数据集来源

课程选题 78 对应“肺炎 X 光图像诊断识别”。本项目使用的公开数据集是 Kermany 等发布的儿童胸部 X 光数据，常用名称为 Chest X-Ray Images (Pneumonia)。

来源链如下：

- 原始数据发布页：Daniel Kermany, Kang Zhang, Michael Goldbaum. Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification. Mendeley Data, Version 2, 2018. DOI: 10.17632/rscbjbr9sj.2
- 论文来源：Kermany D S, Goldbaum M, Cai W, et al. Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell, 2018. DOI: 10.1016/j.cell.2018.02.010
- 常用下载入口：Kaggle 数据集 Chest X-Ray Images (Pneumonia), https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

该数据集来自广州妇女儿童医疗中心的儿科胸部 X 光图像，任务标签为 NORMAL 和 PNEUMONIA；肺炎样本文件名中还可见 bacteria / virus 子类型。项目报告按二分类任务使用，不把子类型作为主分类标签。

下载后应使用数据集中已经给出的 train / val / test 划分，不要重新随机切分，否则结果不能和报告中的测试集指标直接对齐。

## 目录结构

```text
data/raw/chest_xray/
  train/NORMAL/*.jpeg
  train/PNEUMONIA/*.jpeg
  val/NORMAL/*.jpeg
  val/PNEUMONIA/*.jpeg
  test/NORMAL/*.jpeg
  test/PNEUMONIA/*.jpeg
```

放置完成后，在仓库根目录运行：

```bash
python scripts/check_dataset.py --verify-images
```

报告使用的本地可读图像数量为 5856 张，其中 train 5216 张、val 16 张、test 624 张。测试集包含 NORMAL 234 张、PNEUMONIA 390 张。若数据集版本、测试集划分或损坏文件处理方式不同，重新计算得到的评价指标可能不同。

## 外部数据集：RSNA Pneumonia Detection Challenge

项目提升版使用 RSNA Pneumonia Detection Challenge 做外部泛化测试。该数据集原始文件包含 DICOM 图像和标签 CSV，不随本仓库发布。Kaggle 竞赛接口需要认证时，可使用 Hugging Face 镜像 `Baldezo313/rsna-pneumonia-dataset` 下载可复现子集。

```text
data/raw/rsna_pneumonia/
  stage_2_train_images/*.dcm
  # 或 HF 镜像的分目录：
  stage_2_train_images_0/*.dcm
  stage_2_train_images_1/*.dcm
  stage_2_train_images_2/*.dcm
  stage_2_train_labels.csv
  stage_2_detailed_class_info.csv
```

第一版只做图像级二分类。`Target=1` 或存在肺炎框的样本记为 `PNEUMONIA`，`Target=0` 样本记为 `NORMAL`。脚本会按 patientId 固定随机划分为 train 70%、val 10%、test 20%，随机种子为 42：

```bash
python scripts/prepare_rsna_binary.py
python scripts/check_external_dataset.py --data-root data/processed/rsna_binary --verify-images
```

本工作区实际使用 HF 镜像下载扩容子集。由于镜像中部分 patientId 下载失败，先按较大的候选池下载，再用 `--available-only` 只整理本地真实存在的 DICOM：

```bash
python scripts/download_rsna_hf_subset.py \
  --train-per-class 650 \
  --val-per-class 180 \
  --test-per-class 360
python scripts/prepare_rsna_binary.py --images-dir data/raw/rsna_pneumonia --available-only
python scripts/check_external_dataset.py --data-root data/processed/rsna_binary --json-output results/rsna_dataset_check.json --verify-images
```

实际整理得到 1707 张图像。划分为 train NORMAL 560 / PNEUMONIA 492，val 107 / 106，test 215 / 227。摘要文件为 `results/rsna_dataset_summary.json`，可读性检查文件为 `results/rsna_dataset_check.json`。

处理后目录为：

```text
data/processed/rsna_binary/
  train/NORMAL/
  train/PNEUMONIA/
  val/NORMAL/
  val/PNEUMONIA/
  test/NORMAL/
  test/PNEUMONIA/
  metadata.csv
  samples_train.csv
  samples_val.csv
  samples_test.csv
```

RSNA 与 Kermany 的标签来源、图像格式和数据分布不同。报告中应把它写成外部泛化测试，不应和同分布测试混为一谈。

## 备选外部数据集：NIH ChestX-ray14

当 RSNA 数据下载受限时，可以使用 NIH ChestX-ray14 的可控弱标签子集。当前仓库支持使用 `images_001.zip` 和 `Data_Entry_2017_v2020.csv` 构建 `Pneumonia` vs `No Finding` 二分类数据：

```text
data/raw/nih_chestxray14_Data_Entry_2017_v2020.csv
data/raw/nih_chestxray14/images_001.zip
```

生成子集：

```bash
python scripts/prepare_nih_binary.py
```

处理后目录为：

```text
data/processed/nih_binary/
  train/NORMAL/
  train/PNEUMONIA/
  val/NORMAL/
  val/PNEUMONIA/
  test/NORMAL/
  test/PNEUMONIA/
  metadata.csv
  samples_train.csv
  samples_val.csv
  samples_test.csv
```

NIH 标签来自报告文本挖掘，噪声高。本项目只把它作为弱标签外部泛化测试，不能和 Kermany 的儿童胸片二分类标签或 RSNA 的检测挑战标签等同。
