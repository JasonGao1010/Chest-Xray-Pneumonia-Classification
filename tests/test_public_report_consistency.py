import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_html_tracks_latest_strict_and_domain_shift_artifacts():
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")
    strict = json.loads((ROOT / "results/strict_summary.json").read_text(encoding="utf-8"))
    domain = json.loads((ROOT / "results/domain_shift_diagnostic.json").read_text(encoding="utf-8"))

    for group in strict["groups"]:
        for metric in ("balanced_accuracy", "roc_auc", "brier_score"):
            token = f"{group['ensemble'][metric]:.10f}".removeprefix("0")
            assert token in html, (group["dataset"], group["model"], metric, token)

    matched = domain["analyses"]["label_matched"]
    assert f"{matched['overall_roc_auc']:.4f}" in html
    assert f"分组数 {matched['group_count']:,}" in html
    for fold in matched["folds"]:
        token = f"{fold['roc_auc']:.10f}".removeprefix("0")
        assert token in html
    for comparison in strict["paired_comparisons"]:
        delta = comparison["candidate_minus_baseline"]["balanced_accuracy"] * 100
        low, high = comparison["paired_group_bootstrap_95ci"]["balanced_accuracy"]
        for value in (delta, low * 100, high * 100):
            token = f"{value:.4f}".replace("0.", ".").replace("-0.", "-.")
            assert token in html, (comparison["candidate_family"], comparison["dataset"], token)


def test_html_has_no_retired_public_numbers_or_wording():
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")
    for stale in (
        "94.23%", "group=2,217", "RSNA镜像便利子集", "1,707张镜像便利子集",
        "稳健配方", "保守文件名簇", "简单混合", "域均衡混合", "严格基线",
    ):
        assert stale not in html


def test_html_defines_every_public_model_and_method_code():
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")
    for code, meaning in (
        ("DenseNet121", "torchvision DenseNet-121"),
        ("ConvNeXt-Tiny", "torchvision ConvNeXt Tiny"),
        ("ViT-B/16", "torchvision Vision Transformer B/16"),
        ("ERM", "经验风险最小化"),
        ("ERM-Reg", "正则化 ERM"),
        ("JT", "联合训练"),
        ("JT-DBS", "来源均衡联合训练"),
    ):
        assert code in html
        assert meaning in html
    for retired_model_code in ("DN121", "CNXT-T", "ViT-B16"):
        assert retired_model_code not in html
