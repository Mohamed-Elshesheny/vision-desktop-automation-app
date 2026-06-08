from __future__ import annotations

import string
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

Theme = Literal["light", "dark"]

GRID_ROWS = 6
GRID_COLS = 6


def cell_bounds(cell_label: str, img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    if len(cell_label) < 2:
        return None
    row_char, col_str = cell_label[0].upper(), cell_label[1:]
    row_labels = list(string.ascii_uppercase[:GRID_ROWS])
    if row_char not in row_labels or not col_str.isdigit():
        return None
    r, c = row_labels.index(row_char), int(col_str) - 1
    if c < 0 or c >= GRID_COLS:
        return None
    cw, ch = img_w / GRID_COLS, img_h / GRID_ROWS
    return (int(c * cw), int(r * ch), int((c + 1) * cw), int((r + 1) * ch))


def _grid_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_grid_overlay(image: Image.Image) -> Image.Image:
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cw, ch = w / GRID_COLS, h / GRID_ROWS
    font_size = max(14, min(32, int(min(cw, ch) * 0.2)))
    font = _grid_font(font_size)

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            label = f"{string.ascii_uppercase[r]}{c + 1}"
            x0, y0 = int(c * cw), int(r * ch)
            x1, y1 = int((c + 1) * cw), int((r + 1) * ch)
            draw.rectangle([x0, y0, x1, y1], outline="white", width=1)
            draw.text(
                (x0 + 5, y0 + 4),
                label,
                fill="white",
                font=font,
                stroke_width=2,
                stroke_fill="black",
            )

    return img


def draw_anchor_marks(image: Image.Image, points: list[tuple[int, int]]) -> Image.Image:
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _grid_font(14)
    size = 4

    for point_id, (x, y) in enumerate(points):
        draw.line([(x - size, y), (x + size, y)], fill=(255, 0, 255), width=1)
        draw.line([(x, y - size), (x, y + size)], fill=(255, 0, 255), width=1)
        draw.text(
            (x + 3, y - 14),
            str(point_id),
            fill=(255, 0, 255),
            font=font,
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )

    return img


def _to_gray(image: Image.Image) -> np.ndarray:
    arr = np.array(image.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def _system_theme() -> Theme | None:
    import sys
    if sys.platform != "win32":
        return None
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        )
        value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
        winreg.CloseKey(key)
        return "light" if int(value) == 1 else "dark"
    except Exception:  # noqa: BLE001
        return None


def detect_theme(image: Image.Image) -> Theme:
    theme = _system_theme()
    if theme is not None:
        return theme

    gray = _to_gray(image)
    h, w = gray.shape
    patch_h, patch_w = max(1, h // 8), max(1, w // 8)
    corners = [
        gray[0:patch_h, 0:patch_w],
        gray[0:patch_h, w - patch_w : w],
        gray[h - patch_h : h, 0:patch_w],
        gray[h - patch_h : h, w - patch_w : w],
    ]
    mean_luminance = float(np.mean([c.mean() for c in corners]))
    return "dark" if mean_luminance < 128 else "light"


def to_edge_image(image: Image.Image, low: int = 50, high: int = 150) -> np.ndarray:
    gray = _to_gray(image)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(gray, low, high)


def resize_for_vlm(image: Image.Image, max_width: int = 1280, max_height: int = 720) -> Image.Image:
    w, h = image.size
    if w <= max_width and h <= max_height:
        return image
    scale = min(max_width / w, max_height / h)
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size, Image.LANCZOS)
