# 完整复现说明

本仓库的目标是：复现者只需取得仓库代码，并自行下载指定的两套公开数据，即可从数据准备开始重建主要实验和结论，不依赖未公开的检查点或本机隐藏文件。

## 一、软件与硬件环境

参考审计使用 Python 3.13.12，Python 依赖准确版本见 `requirements.txt`。完整流程包含 18 次训练，需要具备足够显存、存储空间和运行时间的 NVIDIA GPU。

不同 GPU、驱动和 CUDA 算子可能带来细微数值差异。因此，验收程序要求数据身份、样本数量和实验矩阵精确一致；对主要模型指标采用预先声明的 5% 相对误差容限。固定逐图预测产生的统计汇总应保持确定性。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

## 二、下载原始数据

### Kermany 胸片数据

从 Mendeley Data version 2 或对应 Kaggle 镜像下载 Chest X-Ray Images (Pneumonia)，并将数据包中的 `train`、`val`、`test` 放到：

```text
data/raw/chest_xray/
```

### RSNA 挑战赛数据

从 Kaggle 的 RSNA Pneumonia Detection Challenge 官方数据页下载训练数据，并整理为：

```text
data/raw/rsna_pneumonia/
  stage_2_train_images/*.dcm
  stage_2_train_labels.csv
  stage_2_detailed_class_info.csv
```

正式复现不使用历史 Hugging Face 镜像。程序从官方完整训练集中抽取冻结的 1,707 个成员，并逐一核对 `data/splits/rsna_available_1707_manifest.csv` 中的 `patientId`、标签、划分和 DICOM SHA-256。任何成员缺失或哈希不符都会终止流程。

## 三、运行完整流程

先查看全部 100 条数据准备、训练、评价、分析和验收命令，不实际执行：

```bash
python scripts/reproduce_all.py --dry-run
```

确认数据目录和计算资源后，运行全部流程：

```bash
python scripts/reproduce_all.py --stage all --device auto
```

所有新产物均写入 `rebuild/`，仓库中的发布证据不会被覆盖。如果数据阶段中断并留下不完整目录，应先检查原因，再明确允许替换 `rebuild/` 内的数据产物：

```bash
python scripts/reproduce_all.py --stage data --force
```

也可以按顺序分阶段执行：

```bash
python scripts/reproduce_all.py --stage data
python scripts/reproduce_all.py --stage experiments --device cuda
python scripts/reproduce_all.py --stage analyze
python scripts/reproduce_all.py --stage verify
```

## 四、验收标准

`scripts/verify_reproduction.py` 同时检查：

- 5,856 张 Kermany 图像全部通过冻结哈希核验；
- 1,707 个 RSNA DICOM 成员全部通过冻结哈希和划分核验；
- 三种架构在两套测试集上的 6 个严格结果组完整存在；
- 三种 DenseNet121 候选方案在两套测试集上的 6 个成对比较完整存在；
- 每个必需实验均包含随机种子 42、43、44；
- 校准、阈值、错误分析和来源可分性等次级证据完整存在；
- 主要集成指标与发布值的相对差异小于 5%。

验收通过后会生成 `rebuild/reproduction_verification.json`，其中状态应为：

```json
{"status": "VERIFIED"}
```

缺少任意必需文件都不会被视为成功的“部分复现”。

## 五、实验矩阵

| 实验族 | 模型 | 随机种子 | 训练数据 | 测试数据 |
|---|---|---|---|---|
| strict | DenseNet121、ConvNeXt-Tiny、ViT-B/16 | 42、43、44 | 分组 Kermany | 分组 Kermany、冻结 RSNA |
| robust | DenseNet121 | 42、43、44 | 分组 Kermany，使用较强增强 | 分组 Kermany、冻结 RSNA |
| mixed_simple | DenseNet121 | 42、43、44 | Kermany 与 RSNA 的训练、验证数据 | 分组 Kermany、冻结 RSNA |
| mixed_domain_balanced | DenseNet121 | 42、43、44 | 混合数据，按来源均衡采样 | 分组 Kermany、冻结 RSNA |

`v1.0.0` 中可下载的 ViT 权重只服务于本地演示，不参与上述严格复现流程。
