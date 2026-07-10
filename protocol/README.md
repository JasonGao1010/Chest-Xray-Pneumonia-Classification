# CXRShift 命名协议

`CXRShift` 是本仓库最终使用的实验身份协议。模型名称保留原名称，本项目自行定义的数据处理和训练方案使用固定英文标识。

## 规范名称

| 类型 | 规范名称 | 含义 |
|---|---|---|
| 模型 | `DenseNet121` | torchvision DenseNet-121 |
| 模型 | `ConvNeXt-Tiny` | torchvision ConvNeXt Tiny |
| 模型 | `ViT-B/16` | torchvision Vision Transformer B/16 |
| 数据 | `Kermany-FG` | Kermany 数据按文件名分组后的固定划分 |
| 数据 | `RSNA-1707` | 固定 1,707 个 RSNA 成员的子集 |
| 训练 | `ERM` | 仅使用 Kermany，以加权交叉熵最小化经验风险 |
| 训练 | `ERM-Reg` | 在 ERM 上增加颜色扰动和 0.1 标签平滑 |
| 训练 | `JT` | 联合使用 Kermany 与 RSNA 训练数据 |
| 训练 | `JT-DBS` | 在 JT 上增加按图像来源均衡的加权采样 |

运行编号采用：

```text
CXRShift__{model}__{recipe}__s{seed}
```

例如：

```text
CXRShift__DenseNet121__JT-DBS__s42
```

新产物文件名由运行编号、数据版本、划分和产物类型组成。历史文件中的 `strict`、`robust`、`mixed_simple`、`mixed_domain_balanced` 作为兼容别名保留，不再用于新产物命名。

## 命名依据

这套名称遵循论文中按训练机制命名的惯例，而不使用“严格”“稳健”“简单”等工程状态词：

- `ERM` 沿用域泛化研究对经验风险最小化基线的常见称呼；
- `JT` 沿用多数据来源联合训练的常见称呼；
- `DBS` 表示 domain-balanced sampling，即按来源均衡采样；
- `Reg` 只表示本项目明确加入的颜色扰动与标签平滑，不暗示新的算法贡献。

参考：[医学影像域泛化强基线](https://arxiv.org/abs/1904.01638)、[胸片多域均衡采样](https://arxiv.org/abs/2112.13734)、[胸片持续学习中的 Joint Training](https://arxiv.org/abs/2001.05922)。这些论文用于校准命名方式，不表示本项目复现了论文中的全部方法。
