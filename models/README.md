# 模型权重说明

GitHub 仓库默认不提交模型权重文件。请从 Release 附件或课程交付压缩包中取得主模型权重，并放置为：

```text
models/vit_b16_best.pt
```

网页演示和主结果复现命令都默认读取这个路径：

```bash
python scripts/serve_patient_app.py --checkpoint models/vit_b16_best.pt --device cpu
```

该权重文件约 1GB，不建议直接提交到普通 Git 仓库。若必须随仓库管理，请使用 Git LFS。
