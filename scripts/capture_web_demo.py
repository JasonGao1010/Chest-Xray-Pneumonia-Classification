#!/usr/bin/env python3
"""Capture two real local-inference states for the public demo figure."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/web_demo_examples.png"),
    )
    return parser.parse_args()


def capture_state(page, image_path: Path) -> Image.Image:
    page.locator("#imageInput").set_input_files(image_path.as_posix())
    page.locator("#predictButton").click()
    page.locator("#statusValue").filter(has_text="推理完成").wait_for(timeout=60_000)
    raw = page.locator(".workbench").screenshot(type="png")
    return Image.open(BytesIO(raw)).convert("RGB")


def main() -> int:
    args = parse_args()
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    normal = PROJECT_ROOT / "sample_images/normal_IM-0001-0001.jpeg"
    pneumonia = PROJECT_ROOT / "sample_images/pneumonia_person100_bacteria_475.jpeg"

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=1)
        page.goto(args.url, wait_until="networkidle")
        page.locator("#serviceStatus.online").wait_for(timeout=30_000)
        normal_panel = capture_state(page, normal)
        page.locator("#clearButton").click()
        pneumonia_panel = capture_state(page, pneumonia)
        browser.close()

    gap = 24
    height = max(normal_panel.height, pneumonia_panel.height)
    combined = Image.new(
        "RGB",
        (normal_panel.width + gap + pneumonia_panel.width, height),
        "white",
    )
    combined.paste(normal_panel, (0, 0))
    combined.paste(pneumonia_panel, (normal_panel.width + gap, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output, dpi=(300, 300), optimize=True)
    try:
        display_output = output.relative_to(PROJECT_ROOT)
    except ValueError:
        display_output = output
    print(f"Wrote {display_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
