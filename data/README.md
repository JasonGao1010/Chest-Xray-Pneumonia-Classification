# 数据集目录说明

完整数据集不随仓库发布。请自行下载 Chest X-Ray Images (Pneumonia) 数据集，并整理为以下目录结构：

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

报告使用的本地可读图像数量为 5856 张。若数据集版本、测试集划分或损坏文件处理方式不同，重新计算得到的评价指标可能不同。
