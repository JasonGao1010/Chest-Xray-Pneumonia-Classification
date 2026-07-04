#!/usr/bin/env python3
"""Run a local web app for single-image chest X-ray classification."""

from __future__ import annotations

import argparse
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from PIL import Image, UnidentifiedImageError
from flask import Flask, jsonify, request, send_file, send_from_directory

from xray_pneumonia.inference import DEFAULT_CHECKPOINT, XRayPredictor, resolve_project_path

WEB_ROOT = PROJECT_ROOT / "web"
SAMPLE_XRAY_CANDIDATES = (
    PROJECT_ROOT / "sample_images/normal_IM-0001-0001.jpeg",
    PROJECT_ROOT / "data/raw/chest_xray/test/NORMAL/IM-0001-0001.jpeg",
)
MAX_UPLOAD_MB = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def create_app(
    predictor: XRayPredictor | None = None,
    checkpoint: Path | str = DEFAULT_CHECKPOINT,
    device: str = "auto",
    threshold: float = 0.5,
) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

    model_predictor = predictor or XRayPredictor(
        checkpoint_path=checkpoint,
        device=device,
        threshold=threshold,
        project_root=PROJECT_ROOT,
    )

    @app.get("/")
    def index():
        return send_from_directory(WEB_ROOT, "index.html")

    @app.get("/app.js")
    def app_js():
        return send_from_directory(WEB_ROOT, "app.js")

    @app.get("/styles.css")
    def styles_css():
        return send_from_directory(WEB_ROOT, "styles.css")

    @app.get("/sample-xray")
    def sample_xray():
        for sample_path in SAMPLE_XRAY_CANDIDATES:
            if sample_path.exists():
                return send_file(sample_path)
        return ("sample image not found", 404)

    @app.get("/health")
    def health():
        return jsonify(
            {
                "ok": True,
                "model": model_predictor.model_name,
                "checkpoint": model_predictor.checkpoint_path.as_posix(),
                "device": str(model_predictor.device_obj),
                "threshold": model_predictor.threshold,
            }
        )

    @app.post("/predict")
    def predict():
        uploaded = request.files.get("image")
        if uploaded is None or not uploaded.filename:
            return jsonify({"ok": False, "error": "请先选择一张胸部 X 光图片。"}), 400

        try:
            payload = uploaded.read()
            with Image.open(BytesIO(payload)) as image:
                result: dict[str, Any] = model_predictor.predict_image(image)
        except UnidentifiedImageError:
            return jsonify({"ok": False, "error": "无法读取该文件，请上传 JPG、PNG 等常见图片格式。"}), 400
        except Exception as exc:
            return jsonify({"ok": False, "error": f"模型推理失败：{exc}"}), 500

        result.update(
            {
                "ok": True,
                "filename": uploaded.filename,
                "clinical_warning": "本结果仅用于课程设计演示，不能替代医生诊断。若有症状或检查异常，请尽快咨询专业医生。",
            }
        )
        return jsonify(result)

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify({"ok": False, "error": f"图片不能超过 {MAX_UPLOAD_MB} MB。"}), 413

    return app


def main() -> int:
    args = parse_args()
    checkpoint = resolve_project_path(args.checkpoint, PROJECT_ROOT)
    app = create_app(
        checkpoint=checkpoint,
        device=args.device,
        threshold=args.threshold,
    )
    print(f"Serving X-ray classifier at http://{args.host}:{args.port}")
    print(f"Checkpoint: {checkpoint.as_posix()}")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
