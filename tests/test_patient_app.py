from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.serve_patient_app import create_app


class FakePredictor:
    model_name = "fake-vit"
    checkpoint_path = Path("fake.pt")
    device_obj = "cpu"
    threshold = 0.5

    def predict_image(self, image):
        assert image.size == (8, 8)
        return {
            "predicted_label": "PNEUMONIA",
            "predicted_label_cn": "疑似肺炎",
            "normal_probability": 0.1,
            "pneumonia_probability": 0.9,
            "probabilities": {"NORMAL": 0.1, "PNEUMONIA": 0.9},
            "threshold": 0.5,
            "confidence": 0.9,
            "model": self.model_name,
            "checkpoint": self.checkpoint_path.as_posix(),
            "device": "cpu",
            "image_size": 224,
        }


def make_png() -> BytesIO:
    handle = BytesIO()
    Image.new("RGB", (8, 8), color=(128, 128, 128)).save(handle, format="PNG")
    handle.seek(0)
    return handle


def test_predict_endpoint_accepts_image_upload():
    app = create_app(predictor=FakePredictor())
    client = app.test_client()

    response = client.post(
        "/predict",
        data={"image": (make_png(), "sample.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["predicted_label"] == "PNEUMONIA"
    assert payload["pneumonia_probability"] == 0.9
    assert "不能替代医生诊断" in payload["clinical_warning"]


def test_predict_endpoint_requires_file():
    app = create_app(predictor=FakePredictor())
    client = app.test_client()

    response = client.post("/predict", data={}, content_type="multipart/form-data")

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["ok"] is False
