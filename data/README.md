# 数据集目录说明

完整数据集不随仓库发布。请自行下载 Chest X-Ray Images (Pneumonia) 数据集，并整理为以下目录结构：

## 数据集来源

本项目使用的公开数据集是 Kermany 等发布的儿童胸部 X 光数据，常用名称为 Chest X-Ray Images (Pneumonia)。

来源链如下：

- 原始数据发布页：Daniel Kermany, Kang Zhang, Michael Goldbaum. [Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification](https://data.mendeley.com/datasets/rscbjbr9sj/2). Mendeley Data, Version 2, 2018. DOI: 10.17632/rscbjbr9sj.2，CC BY 4.0
- 论文来源：Kermany D S, Goldbaum M, Cai W, et al. Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell, 2018. DOI: 10.1016/j.cell.2018.02.010
- 常用下载入口：Kaggle 数据集 Chest X-Ray Images (Pneumonia), https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

该数据集来自广州妇女儿童医疗中心的儿科胸部 X 光图像，任务标签为 NORMAL 和 PNEUMONIA；肺炎样本文件名中还可见 bacteria / virus 子类型。代码按二分类任务使用，不把子类型作为主分类标签。

下载包自带 train / val / test 目录。仓库另提供按文件名派生簇隔离的划分。公开数据未提供可核验的胸片患者映射表。

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

本地可读图像共 5856 张。下载包自带划分为 train 5216 张、val 16 张、test 624 张；其中原始 test 包含 NORMAL 234 张、PNEUMONIA 390 张。若数据版本或损坏文件处理方式不同，重新计算得到的指标可能不同。

正式内部评价使用 `data/processed/kermany_grouped_seed42` 和清单 `data/splits/kermany_grouped_seed42.csv`。该划分采用保守的文件名分组规则，目的只是检测结论对分组策略是否敏感。当前清单为纯 LF 文本，原始字节 SHA-256 为 `a0b117b04248fd475cea737690a3ba3a23d3daee0c0e8289e17545e5f2af9acf`；按 `csv_rows_sorted_canonical_json_v1` 规则得到的语义 SHA-256 为 `819630293dcc8e66884dc211ea84b3fe568235850f3b71a5cf80f8f5ad9cdbcb`。后者不受 CSV 换行格式和行序影响。核验摘要见 `results/kermany_grouped_summary.json`。

两个数据准备脚本默认拒绝覆盖已有输出。需要复建时，优先写入新目录：

```bash
python scripts/prepare_kermany_grouped.py \
  --output-root data/processed/kermany_grouped_seed42_rebuild \
  --manifest data/splits/kermany_grouped_seed42_rebuild.csv \
  --summary results/kermany_grouped_summary_rebuild.json
```

`--force` 会删除命令指定的旧输出，只能在确认这些路径可替换后使用。核对当前冻结清单而不重建图像，可把核验结果写到临时文件：

```bash
python scripts/prepare_kermany_grouped.py \
  --verify-only --verify-files \
  --summary /tmp/kermany_grouped_summary_check.json
```

## 外部数据集：RSNA Pneumonia Detection Challenge

项目使用 RSNA Pneumonia Detection Challenge 来源数据做探索性跨来源压力测试。原始文件包含 DICOM 图像和标签 CSV，不随本仓库发布。Kaggle 竞赛接口需要认证时，历史流程使用了 Hugging Face 镜像 `Baldezo313/rsna-pneumonia-dataset`；该镜像缺少完备数据卡和明确许可说明，不能视为官方分发的等价替代。

官方竞赛数据页：https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data。使用者需分别遵守官方竞赛条款和实际下载源的许可条件。

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

代码沿用二分类目录名：`Target=1` 或存在目标框的样本写入 `PNEUMONIA`，`Target=0` 写入 `NORMAL`。医学语义上，它们应分别理解为“可能肺炎相关阴影阳性”和“目标阴性”，不能解释为确诊肺炎与健康。脚本按 RSNA 提供的 `patientId` 固定随机划分为 train 70%、val 10%、test 20%，随机种子为 42；每张 DICOM 单独做最小--最大归一化后转为 8 位 PNG：

```bash
python scripts/prepare_rsna_binary.py \
  --output-root data/processed/rsna_binary_rebuild \
  --splits-output data/splits/rsna_binary_seed42_rebuild.json \
  --summary-output results/rsna_dataset_summary_rebuild.json \
  --figure-output figures/rsna_class_distribution_rebuild.png
python scripts/check_external_dataset.py --data-root data/processed/rsna_binary_rebuild --verify-images
```

本工作区实际使用 HF 镜像下载扩容子集。由于镜像中部分 patientId 下载失败，先按较大的候选池下载，再用 `--available-only` 只整理本地真实存在的 DICOM：

```bash
python scripts/download_rsna_hf_subset.py \
  --train-per-class 650 \
  --val-per-class 180 \
  --test-per-class 360
python scripts/prepare_rsna_binary.py \
  --images-dir data/raw/rsna_pneumonia --available-only \
  --output-root data/processed/rsna_binary_rebuild \
  --splits-output data/splits/rsna_binary_seed42_rebuild.json \
  --summary-output results/rsna_dataset_summary_rebuild.json \
  --figure-output figures/rsna_class_distribution_rebuild.png
python scripts/check_external_dataset.py \
  --data-root data/processed/rsna_binary_rebuild \
  --json-output results/rsna_dataset_check_rebuild.json \
  --verify-images
```

实际整理得到 1707 张图像。划分为 train NORMAL 560 / PNEUMONIA 492，val 107 / 106，test 215 / 227。摘要文件为 `results/rsna_dataset_summary.json`，可读性检查文件为 `results/rsna_dataset_check.json`。

当前 1707 个成员已冻结在 `data/splits/rsna_available_1707_manifest.csv`。每行记录 patientId、标签、划分、来源记录、原始 DICOM 与处理后 PNG 的仓库相对路径、文件大小和 SHA-256。清单 SHA-256 为 `76ea17aae923954deca87834c1cdca6de0383bee3e461de29840a4564d247ae1`，摘要见 `data/splits/rsna_available_1707_manifest_summary.json`。

两份保留的下载记录可追溯 1573 个成员：最早的小批次记录 88 个，扩展记录新增 1485 个。其余 134 个成员的原始文件真实存在且已固定哈希，但没有对应的保留下载记录，因此清单把来源批次和提供方明确写为 `unknown`。`source_batch` 表示“最早保留下来且列出该成员的下载记录”，不等同于可证明的首次下载时间。生成新清单时运行：

```bash
python scripts/freeze_rsna_manifest.py
```

如果清单已经存在，脚本会停止；只有显式传入 `--force` 才会替换清单和摘要。它不会修改原始或处理后的图像。

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

RSNA 与 Kermany 的标签来源、图像格式和数据分布不同。

## 备选外部数据集：NIH ChestX-ray14

当 RSNA 数据下载受限时，可以使用 NIH ChestX-ray14 的可控弱标签子集。当前仓库支持使用 `images_001.zip` 和 `Data_Entry_2017_v2020.csv` 构建 `Pneumonia` vs `No Finding` 二分类数据：

NIH 公开发布入口：https://nihcc.app.box.com/v/ChestXray-NIHCC。

```text
data/raw/nih_chestxray14_Data_Entry_2017_v2020.csv
data/raw/nih_chestxray14/images_001.zip
```

生成子集：

```bash
python scripts/prepare_nih_binary.py \
  --output-root data/processed/nih_binary_rebuild \
  --summary-output results/nih_dataset_summary_rebuild.json \
  --figure-output figures/nih_class_distribution_rebuild.png
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

NIH 标签来自报告文本挖掘。当前保留的 130 张历史子集存在 8 个跨划分患者编号，不用于严格结果。修正后的准备脚本按八位患者编号分组后再划分；重建应写入新目录，不应覆盖这份历史数据。

当前 Python 和依赖版本已固定在仓库根目录的 `requirements.txt`。其中包含测试依赖 `pytest`；更换 Python、PyTorch 或 CUDA 环境后，应重新运行测试和数据核验，不能直接沿用已有结果。
