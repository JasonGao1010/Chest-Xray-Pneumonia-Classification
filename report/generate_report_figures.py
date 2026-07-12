from __future__ import annotations

import json
import re
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FONT_DIR = Path.home() / ".local/share/fonts/windows-report"
for font_file in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf", "simsun.ttc"):
    font_manager.fontManager.addfont(FONT_DIR / font_file)

# Editable SVG text and embedded TrueType PDF text.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({
    "font.size": 8.5,
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

COLORS = {
    "navy": "#66749A",
    "blue": "#8FA8C9",
    "blue_light": "#C8D5E5",
    "rose": "#C79AA5",
    "rose_light": "#E5CDD3",
    "sage": "#9EB7A3",
    "sage_light": "#D1DED3",
    "sand": "#C8B795",
    "gray": "#9B9B9B",
    "gray_light": "#D8D8D8",
    "ink": "#3F4652",
}


def load(name: str) -> dict:
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def save(fig: plt.Figure, name: str) -> None:
    chinese = re.compile(r"[\u3400-\u9fff]")
    for text in fig.findobj(match=lambda obj: isinstance(obj, matplotlib.text.Text)):
        family = "SimSun" if chinese.search(text.get_text()) else "Times New Roman"
        text.set_fontfamily(family)
    fig.tight_layout(pad=0.8)
    for ext in ("svg", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{name}.png", dpi=360, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def label_bars(ax, bars, digits=3, dy=0.012):
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value + dy, f"{value:.{digits}f}",
                ha="center", va="bottom", fontsize=7.2, color=COLORS["ink"])


strict = load("strict_summary.json")
domain = load("domain_shift_diagnostic.json")
audit = load("integrity_audit_grouped.json")
groups = {(g["dataset"], g["model"]): g for g in strict["groups"]}
comparisons = {(c["candidate_family"], c["dataset"]): c for c in strict["paired_comparisons"]}


# 1. Research workflow (schematic-led composite)
fig, ax = plt.subplots(figsize=(7.1, 1.55))
ax.set_xlim(0, 10); ax.set_ylim(0, 2); ax.axis("off")
items = [
    (0.15, "数据准备", "Kermany-FG\nRSNA-1707", COLORS["blue_light"]),
    (2.15, "完整性审计", "文件名派生簇\n来源标识分组", COLORS["sage_light"]),
    (4.15, "模型训练", "三种架构\n四种训练方案", COLORS["rose_light"]),
    (6.15, "双来源评价", "分类·校准\n错误分析", "#E5DDCC"),
    (8.15, "统计归纳", "三种子集成\n分组自助法", "#D9D8E8"),
]
for x, title, body, color in items:
    patch = FancyBboxPatch((x, 0.38), 1.55, 1.15, boxstyle="round,pad=0.04,rounding_size=0.08",
                           facecolor=color, edgecolor=COLORS["ink"], linewidth=0.7)
    ax.add_patch(patch)
    ax.text(x + 0.775, 1.18, title, ha="center", va="center", fontsize=8.5, weight="bold")
    ax.text(x + 0.775, 0.76, body, ha="center", va="center", fontsize=7.2, linespacing=1.35)
for i in range(4):
    ax.add_patch(FancyArrowPatch((items[i][0] + 1.58, 0.95), (items[i + 1][0] - 0.05, 0.95),
                                 arrowstyle="-|>", mutation_scale=9, color=COLORS["gray"], linewidth=0.9))
save(fig, "fig01_research_workflow")


# 2. Dataset split composition
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.55), sharey=False)
for ax, dataset, title in zip(axes, audit["datasets"], ["Kermany-FG", "RSNA-1707"]):
    splits = ["train", "val", "test"]
    normal = [dataset["counts"][s]["NORMAL"] for s in splits]
    pneumonia = [dataset["counts"][s]["PNEUMONIA"] for s in splits]
    x = np.arange(3)
    ax.bar(x, normal, color=COLORS["blue"], width=0.58, label="NORMAL")
    ax.bar(x, pneumonia, bottom=normal, color=COLORS["rose"], width=0.58, label="PNEUMONIA")
    ax.set_xticks(x, ["训练集", "验证集", "测试集"])
    ax.set_title(title, fontsize=9, weight="bold")
    ax.set_ylabel("图像数量")
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.6)
axes[0].legend(ncol=2, loc="upper right", fontsize=7.2)
save(fig, "fig02_dataset_composition")


# 3. Cross-source balanced accuracy
models = ["densenet121", "convnext_tiny", "vit_b16"]
labels = ["DenseNet121", "ConvNeXt-Tiny", "ViT-B/16"]
src = [groups[("kermany_grouped", m)]["ensemble"]["balanced_accuracy"] for m in models]
tgt = [groups[("rsna", m)]["ensemble"]["balanced_accuracy"] for m in models]
fig, ax = plt.subplots(figsize=(5.9, 2.75))
x = np.arange(3); w = 0.34
b1 = ax.bar(x - w/2, src, w, color=COLORS["blue"], label="源域")
b2 = ax.bar(x + w/2, tgt, w, color=COLORS["rose"], label="跨来源")
label_bars(ax, b1); label_bars(ax, b2)
ax.set_ylim(0, 1.12); ax.set_ylabel("平衡准确率"); ax.set_xticks(x, labels)
ax.axhline(0.5, color=COLORS["gray"], linestyle="--", linewidth=0.8)
ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=7.2)
ax.grid(axis="y", color="#EEEEEE", linewidth=0.6)
save(fig, "fig03_cross_source_balacc")


# 4. Performance-drop slope chart
fig, ax = plt.subplots(figsize=(5.5, 2.7))
for i, (s, t, label, color) in enumerate(zip(src, tgt, labels, [COLORS["navy"], COLORS["sage"], COLORS["rose"]])):
    ax.plot([0, 1], [s, t], marker="o", ms=5, lw=1.7, color=color, label=label)
ax.set_xlim(-0.42, 1.37); ax.set_ylim(0.58, 1.02)
ax.set_xticks([0, 1], ["Kermany-FG", "RSNA-1707"]); ax.set_ylabel("平衡准确率")
ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=6.8)
ax.grid(axis="y", color="#EEEEEE", linewidth=0.6)
save(fig, "fig04_domain_drop")


# 5. Sensitivity/specificity error profile on RSNA
recall = [groups[("rsna", m)]["ensemble"]["recall"] for m in models]
spec = [groups[("rsna", m)]["ensemble"]["specificity"] for m in models]
fig, ax = plt.subplots(figsize=(5.9, 2.75))
b1 = ax.bar(x - w/2, recall, w, color=COLORS["sage"], label="召回率")
b2 = ax.bar(x + w/2, spec, w, color=COLORS["rose"], label="特异度")
label_bars(ax, b1); label_bars(ax, b2)
ax.set_ylim(0, 1.10); ax.set_xticks(x, labels); ax.set_ylabel("指标值")
ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01)); ax.grid(axis="y", color="#EEEEEE", linewidth=0.6)
save(fig, "fig05_rsna_error_profile")


# 6. Training strategy comparison
families = ["strict", "robust", "mixed_simple", "mixed_domain_balanced"]
family_labels = ["ERM", "ERM-Reg", "JT", "JT-DBS"]
base = groups[("rsna", "densenet121")]["ensemble"]
strategy = [base] + [comparisons[(f, "rsna")]["candidate_ensemble"] for f in families[1:]]
bal = [v["balanced_accuracy"] for v in strategy]
rec = [v["recall"] for v in strategy]
spe = [v["specificity"] for v in strategy]
fig, ax = plt.subplots(figsize=(6.5, 2.9))
x4 = np.arange(4); ww = 0.24
ax.bar(x4 - ww, bal, ww, color=COLORS["navy"], label="平衡准确率")
ax.bar(x4, rec, ww, color=COLORS["sage"], label="召回率")
ax.bar(x4 + ww, spe, ww, color=COLORS["rose"], label="特异度")
ax.set_ylim(0, 1.07); ax.set_xticks(x4, family_labels); ax.set_ylabel("指标值")
ax.legend(ncol=3, loc="upper center", fontsize=7.2); ax.grid(axis="y", color="#EEEEEE", linewidth=0.6)
save(fig, "fig06_training_strategies")


# 7. Bootstrap confidence intervals
fig, ax = plt.subplots(figsize=(6.0, 2.25))
entries = [groups[("kermany_grouped", "densenet121")], groups[("rsna", "densenet121")]]
names = ["Kermany-FG", "RSNA-1707"]
for y, (g, name, color) in enumerate(zip(entries, names, [COLORS["blue"], COLORS["rose"]])):
    est = g["ensemble"]["balanced_accuracy"]
    lo, hi = g["ensemble_group_bootstrap_95ci"]["balanced_accuracy"]
    ax.plot([lo, hi], [y, y], color=color, lw=3, solid_capstyle="round")
    ax.plot(est, y, "o", color=COLORS["ink"], ms=5)
    ax.text(hi + 0.008, y, f"{est:.3f} [{lo:.3f}, {hi:.3f}]", va="center", fontsize=7.2)
ax.set_yticks([0, 1], names); ax.invert_yaxis(); ax.set_xlim(0.58, 1.06)
ax.set_xlabel("平衡准确率及95%分组自助置信区间")
ax.grid(axis="x", color="#EEEEEE", linewidth=0.6)
save(fig, "fig07_bootstrap_ci")


# 8. Calibration before/after
cal = load("strict_calibration_rsna_densenet121_seed42.json")
metrics = ["ece", "mce", "brier_score", "negative_log_likelihood"]
metric_labels = ["ECE", "MCE", "Brier", "NLL"]
before = [cal["uncalibrated"][m] for m in metrics]
after = [cal["calibrated"][m] for m in metrics]
fig, ax = plt.subplots(figsize=(5.8, 2.65))
x = np.arange(4)
b1 = ax.bar(x - w/2, before, w, color=COLORS["gray_light"], label="校准前")
b2 = ax.bar(x + w/2, after, w, color=COLORS["blue"], label="校准后")
ax.set_xticks(x, metric_labels); ax.set_ylabel("指标值（越低越好）")
ax.legend(ncol=2, loc="upper left"); ax.grid(axis="y", color="#EEEEEE", linewidth=0.6)
ax.text(0.98, 0.94, f"T = {cal['calibrated']['temperature']:.2f}", transform=ax.transAxes,
        ha="right", va="center", fontsize=7.2, color=COLORS["ink"])
save(fig, "fig08_calibration")


# 9. Source separability under four controls
analysis_labels = ["未调整", "标签匹配", "仅正常", "仅肺炎"]
auc = [
    domain["analyses"]["unadjusted"]["overall_roc_auc"],
    domain["analyses"]["label_matched"]["overall_roc_auc"],
    domain["analyses"]["label_stratified"]["NORMAL"]["overall_roc_auc"],
    domain["analyses"]["label_stratified"]["PNEUMONIA"]["overall_roc_auc"],
]
fig, ax = plt.subplots(figsize=(5.8, 2.65))
bars = ax.bar(np.arange(4), auc, color=[COLORS["gray_light"], COLORS["navy"], COLORS["blue"], COLORS["rose"]], width=0.62)
label_bars(ax, bars, digits=3, dy=0.004)
ax.set_ylim(0.82, 1.0); ax.set_xticks(np.arange(4), analysis_labels); ax.set_ylabel("AUROC")
ax.axhline(0.5, color=COLORS["gray"], linestyle="--", linewidth=0.8)
ax.grid(axis="y", color="#EEEEEE", linewidth=0.6)
save(fig, "fig09_source_separability")


# 10. Descriptive source features (not causal attribution)
features = ["像素均值", "像素标准差", "宽高比", "熵", "中心均值", "边界均值"]
k = np.array([0.480, 0.222, 1.452, 3.050, 0.578, 0.328])
r = np.array([0.503, 0.232, 1.000, 3.137, 0.532, 0.391])
# Normalize each pair to its pair maximum so unlike units can share one descriptive panel.
den = np.maximum(k, r)
fig, ax = plt.subplots(figsize=(6.4, 2.8))
x = np.arange(len(features)); w = 0.34
ax.bar(x - w/2, k/den, w, color=COLORS["blue"], label="Kermany")
ax.bar(x + w/2, r/den, w, color=COLORS["rose"], label="RSNA")
for i, (kv, rv) in enumerate(zip(k, r)):
    ax.text(i - w/2, k[i]/den[i] + 0.025, f"{kv:.3f}", ha="center", fontsize=6.7)
    ax.text(i + w/2, r[i]/den[i] + 0.025, f"{rv:.3f}", ha="center", fontsize=6.7)
ax.set_ylim(0, 1.16); ax.set_xticks(x, features); ax.set_ylabel("组内归一化均值")
ax.legend(ncol=2, loc="upper right"); ax.grid(axis="y", color="#EEEEEE", linewidth=0.6)
save(fig, "fig10_source_features")


# 11. Reproducibility chain
fig, ax = plt.subplots(figsize=(7.1, 1.45))
ax.set_xlim(0, 10); ax.set_ylim(0, 1.6); ax.axis("off")
steps = ["冻结清单", "数据哈希", "十八次训练", "逐图预测", "三种子集成", "分组自助法", "机器验收"]
palette = [COLORS["blue_light"], COLORS["blue_light"], COLORS["sage_light"], COLORS["sage_light"],
           COLORS["rose_light"], COLORS["rose_light"], "#E5DDCC"]
for i, (step, color) in enumerate(zip(steps, palette)):
    x0 = 0.12 + i * 1.4
    ax.add_patch(FancyBboxPatch((x0, 0.47), 1.1, 0.62, boxstyle="round,pad=0.03,rounding_size=0.06",
                                facecolor=color, edgecolor=COLORS["ink"], linewidth=0.65))
    ax.text(x0 + 0.55, 0.78, step, ha="center", va="center", fontsize=7.3)
    if i < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((x0 + 1.11, 0.78), (x0 + 1.37, 0.78), arrowstyle="-|>",
                                     mutation_scale=8, color=COLORS["gray"], linewidth=0.8))
save(fig, "fig11_reproducibility_chain")


def read_ensemble(dataset: str, model: str = "densenet121") -> list[dict]:
    per_seed = []
    for seed in (42, 43, 44):
        path = ROOT / "results" / f"strict_predictions_{dataset}_test_{model}_seed{seed}.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            per_seed.append(list(csv.DictReader(handle)))
    rows = []
    for items in zip(*per_seed):
        paths = {x["path"] for x in items}
        labels_set = {x["true_label"] for x in items}
        if len(paths) != 1 or len(labels_set) != 1:
            raise ValueError("prediction rows are not aligned across seeds")
        probability = float(np.mean([float(x["prob_PNEUMONIA"]) for x in items]))
        rows.append({"path": items[0]["path"], "true_label": items[0]["true_label"],
                     "probability": probability, "predicted_label": "PNEUMONIA" if probability >= 0.5 else "NORMAL"})
    return rows


# 12. Representative radiographs from the two fixed test sets
fig, axes = plt.subplots(2, 4, figsize=(6.8, 3.65))
for row, (root, dataset_name) in enumerate([
    (ROOT / "data/processed/kermany_grouped_seed42/test", "Kermany-FG"),
    (ROOT / "data/processed/rsna_binary/test", "RSNA-1707"),
]):
    selected = []
    for label in ("NORMAL", "PNEUMONIA"):
        paths = sorted((root / label).glob("*"))
        selected.extend([(paths[len(paths)//3], label), (paths[(2*len(paths))//3], label)])
    for col, (path, label) in enumerate(selected):
        ax = axes[row, col]
        ax.imshow(Image.open(path).convert("L"), cmap="gray")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, fontsize=7.2, pad=2)
        for spine in ax.spines.values(): spine.set_color("#B8B8B8"); spine.set_linewidth(0.6)
    axes[row, 0].set_ylabel(dataset_name, fontsize=8.2, weight="bold", labelpad=8)
save(fig, "fig12_representative_cxr")


# 13. Ensemble confusion matrices, reconstructed from the released predictions
fig, axes = plt.subplots(1, 2, figsize=(5.7, 2.55))
for ax, token, title in zip(axes, ["kermany_grouped", "rsna"], ["Kermany-FG", "RSNA-1707"]):
    rows = read_ensemble(token)
    matrix = np.zeros((2, 2), dtype=int)
    index = {"NORMAL": 0, "PNEUMONIA": 1}
    for row in rows:
        matrix[index[row["true_label"]], index[row["predicted_label"]]] += 1
    im = ax.imshow(matrix, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("muted_blue", ["#F1F3F6", COLORS["blue"]]))
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=10,
                    color="white" if matrix[i, j] > matrix.max() * 0.55 else COLORS["ink"])
    ax.set_xticks([0, 1], ["NORMAL", "PNEUMONIA"], rotation=15)
    ax.set_yticks([0, 1], ["NORMAL", "PNEUMONIA"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=9, weight="bold")
    for spine in ax.spines.values(): spine.set_visible(False)
save(fig, "fig13_ensemble_confusion")


# 14. Reliability curves before and after temperature scaling
fig, ax = plt.subplots(figsize=(4.7, 3.05))
for key, label, color, marker in [
    ("uncalibrated", "Before calibration", COLORS["gray"], "o"),
    ("calibrated", "After calibration", COLORS["navy"], "s"),
]:
    bins = [b for b in cal[key]["bins"] if b["count"] > 0]
    ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins],
            color=color, marker=marker, ms=4, lw=1.5, label=label)
ax.plot([0.5, 1.0], [0.5, 1.0], linestyle="--", color=COLORS["gray_light"], lw=1.0, label="Ideal")
ax.set_xlim(0.48, 1.01); ax.set_ylim(0.28, 1.02)
ax.set_xlabel("Mean confidence"); ax.set_ylabel("Empirical accuracy")
ax.legend(loc="lower right", fontsize=7.0); ax.grid(color="#EEEEEE", linewidth=0.6)
save(fig, "fig14_reliability_curve")


# 15. Error examples and Grad-CAM evidence plate
rsna_rows = read_ensemble("rsna")
fp = sorted([r for r in rsna_rows if r["true_label"] == "NORMAL" and r["predicted_label"] == "PNEUMONIA"],
            key=lambda r: r["probability"], reverse=True)[:2]
fn = sorted([r for r in rsna_rows if r["true_label"] == "PNEUMONIA" and r["predicted_label"] == "NORMAL"],
            key=lambda r: r["probability"])[:2]
grad = Image.open(ROOT / "figures/gradcam_rsna_positive_cases_densenet121.png").convert("RGB")
gw, gh = grad.size
grad_crops = [grad.crop((int(i*gw/4)+20, 55, int((i+1)*gw/4)-20, int(gh/3)-15)) for i in range(4)]
fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.0))
for i, row in enumerate(fp + fn):
    ax = axes[0, i]
    image_path = Path(row["path"])
    if not image_path.is_absolute():
        image_path = ROOT / "data/processed/rsna_binary" / image_path
    ax.imshow(Image.open(image_path).convert("L"), cmap="gray")
    ax.set_title(f"{'FP' if i < 2 else 'FN'}  P={row['probability']:.3f}", fontsize=7.1, pad=2)
    ax.set_xticks([]); ax.set_yticks([])
for i, crop in enumerate(grad_crops):
    ax = axes[1, i]; ax.imshow(crop); ax.set_title(f"Grad-CAM {i+1}", fontsize=7.1, pad=2)
    ax.set_xticks([]); ax.set_yticks([])
for ax in axes.flat:
    for spine in ax.spines.values(): spine.set_color("#B8B8B8"); spine.set_linewidth(0.55)
axes[0, 0].set_ylabel("错误样本", fontsize=8.0, labelpad=7)
axes[1, 0].set_ylabel("响应示例", fontsize=8.0, labelpad=7)
save(fig, "fig15_error_gradcam")


print(f"generated 15 figure sets in {OUT}")
