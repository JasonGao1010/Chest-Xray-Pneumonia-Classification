"""Shared typography settings for report figures.

The report figure contract is intentionally narrow:
- confusion matrices show the operating-point error structure;
- error grids show the fixed, traceable FP/FN sample selection;
- plots must preserve source values and image pixels while using Times New Roman labels.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_TIMES_NEW_ROMAN_PATHS = (
    Path("/mnt/c/Windows/Fonts/times.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"),
)


def _times_new_roman_path() -> Path | None:
    """Return an optional local Times New Roman font without requiring WSL."""
    configured = os.environ.get("XRAY_TIMES_NEW_ROMAN")
    candidates = (Path(configured).expanduser(),) if configured else DEFAULT_TIMES_NEW_ROMAN_PATHS
    return next((path for path in candidates if path.is_file()), None)


def configure_report_figure_typography(matplotlib_module: Any) -> str:
    """Register Times New Roman when available and configure publication exports."""
    from matplotlib import font_manager

    family = "Times New Roman"
    font_path = _times_new_roman_path()
    if font_path is not None:
        font_manager.fontManager.addfont(font_path.as_posix())
        family = font_manager.FontProperties(fname=font_path.as_posix()).get_name()

    matplotlib_module.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [family, "Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    return family
