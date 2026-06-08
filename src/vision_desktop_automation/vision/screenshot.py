from __future__ import annotations

import base64
import io
import time

import mss
from loguru import logger
from PIL import Image

_capture_count = 0


def detect_screen_size() -> tuple[int, int]:
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        return monitor["width"], monitor["height"]


def capture_desktop() -> Image.Image:
    global _capture_count
    _capture_count += 1
    seq = _capture_count

    t0 = time.perf_counter()
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        raw = sct.grab(monitor)
    img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.debug(
        f"  screenshot #{seq}: {img.width}×{img.height}px "
        f"captured in {elapsed_ms:.1f}ms"
    )
    return img


def pil_to_png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
