# 工作记录

## 2026-07-10 严格审计与全量重建

- 对全部数据、40 份评价结果、权重、报告和 Git 状态进行审计。
- 发现 Kermany 肺炎类 170 个 subject ID 跨 official train/test；无精确图像跨 split、无标签冲突，旧指标复算一致。
- 发现 `configs/vit_b16.yaml` 与发布 checkpoint 设置漂移；已恢复 learning rate 5e-5、batch 16、AMP。
- 新增完整性审计、患者级划分、来源诊断、多 seed 聚合和冻结阈值工具；自动测试增至 17 项。
- 生成患者级 Kermany 划分：4107/579/1170 张，manifest SHA256 固定。
- 完成 DenseNet121、ConvNeXt-Tiny、ViT-B/16 各 3 seeds 严格重训和双域评价。
- 完成稳健 DenseNet、简单混合、域均衡各 3 seeds；简单混合达到最佳双域折中。
- 修复温度搜索上限截断问题，RSNA Kermany-only 温度扩展搜索后均不在边界。
- 覆盖重写《项目提升计划.md》《报告撰写规划.md》和完整 LaTeX 报告。
- 新报告进一步按硕士学位论文常见板块重组为8章，补充符号表、研究现状、总体需求和系统测试设计；编译为70页，连续正文51页，无LaTeX undefined/overfull警告。

## 2026-07-10 最终落实

- 补跑 DenseNet121 全模型低学习率 RSNA 适配，并同时评价 RSNA test 与 Kermany test。
- 结果为负：RSNA F1 0.7743，Kermany F1 0.9087，没有双域共同提升，保留在完整矩阵。
- 新增 RSNA 概率分布图生成工具并生成 ViT-B/16、DenseNet121 图表。
- 报告补入跨数据集、复现、限制和结论章节，加入关键图表证据与复现验收表。
- LaTeX 独立编译为 51 页 PDF；自动测试 14 项通过。
