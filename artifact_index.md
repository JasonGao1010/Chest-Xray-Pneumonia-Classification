# 产物索引

| 严格版产物 | 用途 | 主要位置 | 状态与边界 |
|---|---|---|---|
| 完整性审计 | 患者交叉、重复、标签和指标复算 | `results/integrity_audit.json`, `reports/audit_review.md` | 原 Kermany 门禁失败是已确认事实 |
| 患者级划分 | 无 subject 交叉的内部协议 | `data/splits/kermany_grouped_seed42.csv`, `results/kermany_grouped_summary.json` | 4107/579/1170 张 |
| 严格三模型汇总 | 3 seeds 与患者 bootstrap | `results/strict_summary.json`, `results/strict_summary.csv` | 内部高分，RSNA 明显下降 |
| 来源诊断 | 简单图像统计区分来源 | `results/domain_shift_diagnostic.json` | AUC 0.9551，不作因果证明 |
| 稳健训练 | 强度扰动+标签平滑 | `results/robust_summary.json` | 固定阈值改善，外部 AUC 下降 |
| 跨域训练 | 简单混合与域均衡 3 seeds | `results/mixed_strict_summary.json` | 简单混合为当前最佳 |
| 正式报告 | 硕士论文式8章高密度正文与复现附录 | `reports/course_report.pdf`, `reports/thesis/main.tex` | 70页，连续正文51页 |

| 产物族 | 用途 | 主要位置 | 验证与限制 |
|---|---|---|---|
| 冻结基线 | Kermany 主结果复现 | `results/baseline_manifest.json` | 主混淆矩阵与权重哈希固定 |
| RSNA 数据 | 外部验证与校准 | `results/rsna_dataset_summary.json` | 1707 张；test 215/227；非完整挑战数据 |
| 模型比较 | 三结构双域比较 | `results/experiment_comparison.csv` | 指标来自固定 test |
| 泛化策略 | 六种 DenseNet121 路线 | `results/eval_*expanded*.json` | 固定 seed=42，无置信区间 |
| 校准 | ECE、Brier、温度缩放 | `results/calibration_comparison.csv` | 温度只从 validation 学习 |
| 错误与解释 | FP/FN、高置信错误、Grad-CAM | `results/error_*`, `figures/gradcam_*` | Grad-CAM 不作病灶定位 |
| 正式报告 | 旧版交付记录，已由严格版替代 | `reports/course_report.pdf`, `reports/thesis/main.tex` | 当前为 70 页严格版 |
