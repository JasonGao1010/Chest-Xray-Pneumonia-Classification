# 当前任务状态

- 阶段：strict final package
- 核心修正：发现 Kermany 官方 train/test 有 170 个肺炎 subject ID 交叉；旧结果降级为公开目录复现。
- 新协议：患者级 train/validation/locked test 为 4107/579/1170 张，患者、SHA256 和标签冲突门禁通过。
- 已完成：三模型×3 seeds 严格基线、RSNA 外部评价、患者 bootstrap、来源诊断、稳健训练、简单混合、域均衡、目标域校准、validation 冻结阈值和错误分析。
- 当前最佳：简单混合 DenseNet；RSNA BAcc 0.7645±0.0095，内部 BAcc 0.9684±0.0040。
- 报告：新 LaTeX 报告 70 页，参照硕士学位论文结构组织为8章，连续正文51页，不靠图片附录填页。
- 边界：简单混合使用 RSNA 标签，不属于未知域泛化；NIH 仍是 130 张弱标签敏感性分析；不构成临床验证。
- 待办：最终一致性审计、Git 提交与远端同步。
