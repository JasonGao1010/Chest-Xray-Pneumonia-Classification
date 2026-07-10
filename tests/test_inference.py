from __future__ import annotations

from types import SimpleNamespace

import pytest
import torchvision.models

from xray_pneumonia.inference import create_classifier_model


@pytest.mark.parametrize(
    ("model_name", "constructor_name"),
    [
        ("torchvision:densenet121", "densenet121"),
        ("densenet121", "densenet121"),
        ("torchvision:convnext_tiny", "convnext_tiny"),
        ("convnext_tiny", "convnext_tiny"),
        ("torchvision:vit_b_16", "vit_b_16"),
        ("vit_b16", "vit_b_16"),
    ],
)
def test_create_classifier_model_dispatches_to_torchvision(
    monkeypatch,
    model_name,
    constructor_name,
):
    sentinel = object()
    calls = []

    def fake_constructor(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(torchvision.models, constructor_name, fake_constructor)
    timm_module = SimpleNamespace(
        create_model=lambda *_args, **_kwargs: pytest.fail("timm fallback was used")
    )

    model = create_classifier_model(model_name, num_classes=2, timm_module=timm_module)

    assert model is sentinel
    assert calls == [{"weights": None, "num_classes": 2}]


def test_create_classifier_model_keeps_timm_fallback():
    sentinel = object()
    calls = []

    def create_model(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    model = create_classifier_model(
        "tf_efficientnetv2_s.in1k",
        num_classes=2,
        timm_module=SimpleNamespace(create_model=create_model),
    )

    assert model is sentinel
    assert calls == [
        (("tf_efficientnetv2_s.in1k",), {"pretrained": False, "num_classes": 2})
    ]
