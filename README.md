# CXRShift：胸片肺炎二分类与跨来源评估

`CXRShift` 是一个基于 PyTorch 的胸片二分类与跨来源评估项目，覆盖数据完整性审计、模型训练、概率校准、跨来源压力测试和本地研究演示。

本项目仅用于研究与教学，不是医疗器械，输出结果不得用于诊断或患者风险判断。

## 主要结论

在单一公开数据集上取得的高性能，不能稳定迁移到不同来源的数据。DenseNet121、ConvNeXt-Tiny 和 ViT-B/16 均使用随机种子 42、43、44 训练，再对三个模型的阳性概率取平均。三种架构在 Kermany 保留测试集上表现较好，但在 RSNA 固定子集上均明显下降：

| 模型 | Kermany 平衡准确率 | RSNA 平衡准确率 | RSNA 特异度 |
|---|---:|---:|---:|
| DenseNet121 | 0.9765 | 0.6591 | 0.3535 |
| ConvNeXt-Tiny | 0.9727 | 0.6695 | 0.4140 |
| ViT-B/16 | 0.9817 | 0.6594 | 0.3628 |

Kermany 测试集包含 1,170 张图像，按照较严格的文件名规则归为 624 组。RSNA 测试集包含 442 张图像，对应 442 个 `patientId`。两套数据在人群、标签、图像格式和分布上均有差异，因此这里衡量的是跨来源压力，而非临床外部验证。

简单的有监督混合训练将 DenseNet121 在 RSNA 上的集成平衡准确率从 0.6591 提高到 0.7813，同时召回率从 0.9648 降至 0.7533，Kermany 性能也略有下降。这说明训练改变了工作点和错误取舍，不能据此声称模型已经泛化到未知医院。成对分组自助法结果见 [`results/strict_summary.json`](results/strict_summary.json)。

## 证据边界

- Kermany 的公开元数据没有经过核验的患者映射。项目只能依据文件名进行分组，不能表述为已确认的患者级隔离。
- RSNA 标签对应挑战赛目标征象，不是经过临床复核的诊断结论。
- 当前 1,707 张 RSNA 工作子集最初来自第三方镜像，其中 134 个成员缺少保留的历史下载记录；对外复现改为从官方完整数据中按冻结 `patientId` 和 SHA-256 精确抽取。
- 早期 NIH 小子集使用较弱的报告文本标签，并存在跨划分患者重叠，因此不进入严格结果。
- 自助区间衡量固定预测下的测试样本不确定性，不包含重新训练更多随机种子带来的不确定性。

数据来源、许可和划分语义见 [`data/README.md`](data/README.md)。

## 仓库结构

```text
configs/        训练配置
data/           数据说明和冻结划分清单
figures/        主要结果图
models/         模型权重来源与下载说明
results/        指标、逐图预测和审计记录
report/         冻结研究报告、可编辑源文件和配套图表
scripts/        数据准备、训练、评价和审计程序
src/            可复用 Python 包
tests/          自动测试
web/            本地推理界面和离线结果报告
```

主要证据入口包括：

- [`results/strict_summary.json`](results/strict_summary.json)：三随机种子集成、分组自助区间和成对比较；
- [`results/integrity_audit_grouped.json`](results/integrity_audit_grouped.json)：Kermany 分组数据和 RSNA 数据的完整性审计；
- [`results/domain_shift_diagnostic.json`](results/domain_shift_diagnostic.json)：控制诊断标签构成后的来源可分性分析；
- [`scripts/README.md`](scripts/README.md)：各阶段命令和输出约定；
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)：从原始数据到主表的完整复现流程和机器验收标准。
- [`protocol/README.md`](protocol/README.md)：项目统一使用的数据、模型、训练方案、运行编号和产物命名规则。
- [`report/main.pdf`](report/main.pdf)：冻结版完整研究报告；[`report/main.tex`](report/main.tex) 为可编辑源文件。

## 安装与测试

最终审计使用 Python 3.13.12，准确依赖版本记录在 `requirements.txt`：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

当前测试套件包含 51 项测试。GPU 训练还需要相容的 NVIDIA 驱动和 CUDA 运行环境。

## 完整复现

按照数据说明下载 Kermany 和官方 RSNA 数据后，可以先查看完整命令计划，再执行全部数据准备、18 次训练、双数据集评价、校准、阈值分析、错误分析和验收：

```bash
python scripts/reproduce_all.py --dry-run
python scripts/reproduce_all.py --stage all --device auto
```

新产物统一写入 `rebuild/`，不会覆盖仓库中的发布证据。只有 `rebuild/reproduction_verification.json` 显示 `"status": "VERIFIED"`，才表示数据身份、实验矩阵和主要指标均通过验收。完整说明见 [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)。

## 数据准备与单次训练

原始和处理后图像不会进入 Git。数据下载入口、许可、目录结构、冻结清单和非覆盖式重建命令见 [`data/README.md`](data/README.md)。单次 DenseNet121 训练示例：

```bash
python scripts/train.py \
  --config configs/DenseNet121__ERM.yaml \
  --data-root data/processed/kermany_grouped_seed42 \
  --results-dir rebuild/results \
  --checkpoints-dir rebuild/checkpoints \
  --seed 42 \
  --json-output rebuild/results/CXRShift__DenseNet121__ERM__s42__training.json
```

完整的多模型、多种子和分析流程由总复现入口负责，避免手工遗漏实验。

## 本地演示

`v1.0.0` Release 中的 ViT-B/16 权重来自早期 Kermany 官方目录划分，只用于本地研究演示，不属于最新严格三种子结果。权重来源和 SHA-256 见 [`models/README.md`](models/README.md)。

```bash
python scripts/serve_patient_app.py \
  --checkpoint models/vit_b16_best.pt --device cpu --threshold 0.5
```

浏览器访问 `http://127.0.0.1:7860`。界面展示的是模型分数与阈值判定，不代表临床性能。

## 许可

项目代码采用 [MIT License](LICENSE)。数据集、预训练权重和第三方资产仍受各自许可及使用条款约束。
