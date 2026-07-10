# 产物索引

| 产物族 | 用途 | 主要位置 | 验证与限制 |
|---|---|---|---|
| 冻结基线 | Kermany 主结果复现 | `results/baseline_manifest.json` | 主混淆矩阵与权重哈希固定 |
| RSNA 数据 | 外部验证与校准 | `results/rsna_dataset_summary.json` | 1707 张；test 215/227；非完整挑战数据 |
| 模型比较 | 三结构双域比较 | `results/experiment_comparison.csv` | 指标来自固定 test |
| 泛化策略 | 六种 DenseNet121 路线 | `results/eval_*expanded*.json` | 固定 seed=42，无置信区间 |
| 校准 | ECE、Brier、温度缩放 | `results/calibration_comparison.csv` | 温度只从 validation 学习 |
| 错误与解释 | FP/FN、高置信错误、Grad-CAM | `results/error_*`, `figures/gradcam_*` | Grad-CAM 不作病灶定位 |
| 正式报告 | 最终课程报告 | `reports/course_report.tex`, `reports/course_report.pdf` | XeLaTeX 独立编译，51 页 |
