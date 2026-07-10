# 模型权重说明

GitHub 仓库默认不提交模型权重文件。请从 v1.0.0 Release 附件下载主模型权重，并放置为：

```text
models/vit_b16_best.pt
```

Release 地址：

```text
https://github.com/JasonGao1010/Chest-Xray-Pneumonia-Classification/releases/tag/v1.0.0
```

权重文件 SHA256：

```text
a15ed64c740f521ed176dfc381c725e7a52d5ec235365e883d049828b277bf0f
```

网页演示默认读取这个路径。该权重来自早期 Kermany 官方目录划分训练，不是 `results/strict_*` 对应的三随机种子模型：

```bash
python scripts/serve_patient_app.py --checkpoint models/vit_b16_best.pt --device cpu
```

该权重文件约 1GB，不建议直接提交到普通 Git 仓库。若必须随仓库管理，请使用 Git LFS。
