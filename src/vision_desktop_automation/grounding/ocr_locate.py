from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from loguru import logger
from PIL import Image

from ..models import BoundingBox, IconCandidate
from .base import is_exact_icon_label

_ICON_OFFSET_PX = 32
_MIN_CLICK_Y = 20
_MIN_LABEL_TOP_Y = 24
_TASKBAR_Y_RATIO = 0.90
_MIN_LABEL_W, _MAX_LABEL_W = 20, 200
_MIN_LABEL_H, _MAX_LABEL_H = 8, 40
_MAX_LABEL_BOTTOM_RATIO = 0.88
_COLUMN_WIDTH_RATIO = 0.26
_COLUMN_X_TOL = 48
_LINE_GAP_PX = 22
_CONTINUATION_START = ("-", "–", "+", "(")


def _is_caption_continuation(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if t.startswith(_CONTINUATION_START):
        return True
    tl = t.lower()
    return tl in {"copy", "copy)"} or tl.startswith("copy ")


@dataclass
class _OcrEntry:
    text: str
    box: list
    score: float
    cx: int
    cy: int
    x1: int
    y1: int
    x2: int
    y2: int


@lru_cache(maxsize=1)
def _ocr_engine():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return None
    return RapidOCR()


def warmup_ocr() -> None:
    if sys.platform != "win32":
        return
    engine = _ocr_engine()
    if engine is None:
        return
    try:
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        engine(dummy)
        logger.debug("  OCR engine warmed up.")
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"  OCR warmup skipped: {exc!r}")


def _parse_entries(result: list, x_offset: int = 0, y_offset: int = 0) -> list[_OcrEntry]:
    parsed: list[_OcrEntry] = []
    for entry in result:
        if len(entry) < 3:
            continue
        box, text, score = entry[0], str(entry[1]).strip(), float(entry[2])
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        parsed.append(
            _OcrEntry(
                text=text,
                box=box,
                score=score,
                cx=int(sum(xs) / len(xs)) + x_offset,
                cy=int(sum(ys) / len(ys)) + y_offset,
                x1=int(min(xs)) + x_offset,
                y1=int(min(ys)) + y_offset,
                x2=int(max(xs)) + x_offset,
                y2=int(max(ys)) + y_offset,
            )
        )
    return parsed


def _same_icon_column(a: _OcrEntry, b: _OcrEntry) -> bool:
    return abs(a.cx - b.cx) <= _COLUMN_X_TOL


def merged_label_text(entry: _OcrEntry, parsed: list[_OcrEntry]) -> str:
    lines: list[tuple[int, str]] = [(entry.y1, entry.text)]
    for other in parsed:
        if other is entry or not _same_icon_column(entry, other):
            continue
        gap = other.y1 - entry.y2
        if 0 <= gap <= _LINE_GAP_PX and _is_caption_continuation(other.text):
            lines.append((other.y1, other.text))
    lines.sort(key=lambda item: item[0])
    return " ".join(text for _, text in lines).strip()


def _is_plausible_desktop_caption(entry: _OcrEntry, sh: int) -> bool:
    return _is_plausible_caption(entry, sh, crop_top_y=0, screen_h=sh)


def _is_plausible_caption(
    entry: _OcrEntry,
    region_h: int,
    *,
    crop_top_y: int = 0,
    screen_h: int | None = None,
) -> bool:
    w, h = entry.x2 - entry.x1, entry.y2 - entry.y1
    if w < _MIN_LABEL_W or w > _MAX_LABEL_W or h < _MIN_LABEL_H or h > _MAX_LABEL_H:
        return False
    if screen_h is not None:
        screen_y2 = crop_top_y + entry.y2
        taskbar_y = int(screen_h * _TASKBAR_Y_RATIO)
        if screen_y2 >= taskbar_y:
            return True
        if screen_y2 > int(screen_h * _MAX_LABEL_BOTTOM_RATIO):
            return False
        return True
    if entry.y2 > int(region_h * _MAX_LABEL_BOTTOM_RATIO):
        return False
    return True


def _icon_click_from_label_box(
    box: list,
    icon_offset: int,
    *,
    x_offset: int = 0,
    y_offset: int = 0,
    crop_top_y: int = 0,
    screen_height: int | None = None,
) -> tuple[int, int] | None:
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    cx = int(sum(xs) / len(xs)) + x_offset
    y_top = int(min(ys)) + y_offset
    y_center = int(sum(ys) / len(ys)) + y_offset
    screen_top_y = y_top if y_offset else crop_top_y + int(min(ys))
    screen_center_y = y_center if y_offset else crop_top_y + int(sum(ys) / len(ys))
    if screen_top_y < _MIN_LABEL_TOP_Y and (
        screen_height is None or screen_center_y < int(screen_height * _TASKBAR_Y_RATIO)
    ):
        return None
    if screen_height is not None and screen_center_y >= int(screen_height * _TASKBAR_Y_RATIO):
        if y_offset:
            return cx, y_center
        return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
    cy = max(_MIN_CLICK_Y, y_top - icon_offset)
    return cx, cy


def _box_center_above_label(
    box: list,
    icon_offset: int,
    *,
    x_offset: int = 0,
    y_offset: int = 0,
    screen_height: int | None = None,
) -> tuple[int, int, BoundingBox] | None:
    click = _icon_click_from_label_box(
        box,
        icon_offset,
        x_offset=x_offset,
        y_offset=y_offset,
        screen_height=screen_height,
    )
    if click is None:
        return None
    cx, cy = click
    half = max(28, icon_offset // 2 + 8)
    return cx, cy, BoundingBox(
        x1=cx - half,
        y1=cy - half,
        x2=cx + half,
        y2=cy + half,
    )


def _pick_best_from_results(
    result: list,
    target_label: str,
    *,
    min_confidence: float,
    icon_offset_px: int,
    sw: int,
    sh: int,
    x_offset: int = 0,
    y_offset: int = 0,
) -> IconCandidate | None:
    taskbar_y = int(sh * _TASKBAR_Y_RATIO)
    parsed = _parse_entries(result, x_offset, y_offset)
    matches: list[tuple[_OcrEntry, str]] = []

    for entry in parsed:
        if entry.score < min_confidence:
            continue
        if not _is_plausible_desktop_caption(entry, sh):
            logger.debug(
                f"  OCR: reject '{entry.text}' — implausible caption box "
                f"({entry.x2 - entry.x1}x{entry.y2 - entry.y1} at y={entry.y1})."
            )
            continue
        merged = merged_label_text(entry, parsed)
        if not is_exact_icon_label(merged, target_label):
            if is_exact_icon_label(entry.text, target_label) and merged != entry.text:
                logger.debug(
                    f"  OCR: reject partial read '{entry.text}' "
                    f"(merged caption '{merged}')."
                )
            continue
        matches.append((entry, merged))

    if not matches:
        return None

    matches.sort(key=lambda item: (item[0].y1, -item[0].score))
    chosen, merged = matches[0]
    if len(matches) > 1:
        logger.debug(
            f"  OCR: {len(matches)} exact '{target_label}' caption(s) — "
            f"picking topmost at y={chosen.y1}."
        )

    mapped = _box_center_above_label(
        chosen.box, icon_offset_px, x_offset=x_offset, y_offset=y_offset, screen_height=sh
    )
    if mapped is None:
        logger.debug(
            f"  OCR: reject '{merged}' — label too close to screen top (y={chosen.y1})."
        )
        return None
    cx, cy, bbox = mapped
    if cy >= taskbar_y:
        return None
    bbox = bbox.clamped(sw, sh)

    return IconCandidate(
        label=target_label,
        box=bbox,
        confidence=min(1.0, chosen.score),
        is_target_match=True,
        reasoning=f"OCR caption '{merged}' score={chosen.score:.2f}",
        source="ocr",
        click_xy=(cx, cy),
        force_verify=True,
    )


def _run_ocr(engine, image: Image.Image) -> list | None:
    arr = np.array(image.convert("RGB"))
    try:
        result, _elapsed = engine(arr)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"  OCR engine failed: {exc!r}")
        return None
    return result or None


def locate_by_label_ocr(
    screenshot: Image.Image,
    target_label: str,
    *,
    min_confidence: float = 0.55,
    icon_offset_px: int = _ICON_OFFSET_PX,
    column_scan: bool = True,
    scan_right_column: bool = False,
) -> IconCandidate | None:
    if sys.platform != "win32":
        return None

    engine = _ocr_engine()
    if engine is None:
        logger.debug("  OCR fast path unavailable (install rapidocr-onnxruntime).")
        return None

    sw, sh = screenshot.size

    if column_scan:
        strip_w = max(200, int(sw * _COLUMN_WIDTH_RATIO))
        columns = [0]
        if scan_right_column:
            columns.append(sw - strip_w)
        for x1 in columns:
            crop = screenshot.crop((x1, 0, x1 + strip_w, sh))
            result = _run_ocr(engine, crop)
            if not result:
                continue
            hit = _pick_best_from_results(
                result,
                target_label,
                min_confidence=min_confidence,
                icon_offset_px=icon_offset_px,
                sw=sw,
                sh=sh,
                x_offset=x1,
            )
            if hit is not None:
                logger.info(
                    f"  OCR fast path: exact label '{target_label}' at {hit.center} "
                    f"(conf={hit.confidence:.2f}, column x={x1})"
                )
                return hit

    taskbar_y = int(sh * _TASKBAR_Y_RATIO)
    strip = screenshot.crop((0, taskbar_y, sw, sh))
    result = _run_ocr(engine, strip)
    if result:
        hit = _pick_taskbar_from_results(
            result,
            target_label,
            min_confidence=min_confidence,
            sw=sw,
            sh=sh,
            y_offset=taskbar_y,
            icon_offset_px=icon_offset_px,
        )
        if hit is not None:
            logger.info(
                f"  OCR taskbar: exact label '{target_label}' at {hit.center} "
                f"(conf={hit.confidence:.2f})"
            )
            return hit

    logger.debug(f"  OCR: no exact match for '{target_label}' in desktop columns.")
    return None


def _pick_taskbar_from_results(
    result: list,
    target_label: str,
    *,
    min_confidence: float,
    sw: int,
    sh: int,
    y_offset: int = 0,
    icon_offset_px: int = _ICON_OFFSET_PX,
) -> IconCandidate | None:
    taskbar_y = int(sh * _TASKBAR_Y_RATIO)
    parsed = _parse_entries(result, y_offset=y_offset)
    matches: list[tuple[_OcrEntry, str]] = []

    for entry in parsed:
        if entry.score < min_confidence:
            continue
        if entry.cy < taskbar_y:
            continue
        merged = merged_label_text(entry, parsed)
        if not is_exact_icon_label(merged, target_label):
            continue
        matches.append((entry, merged))

    if not matches:
        return None

    matches.sort(key=lambda item: (-item[0].score, item[0].x1))
    chosen, merged = matches[0]
    cx, cy = chosen.cx, chosen.cy
    half = max(20, icon_offset_px // 2)
    bbox = BoundingBox(x1=cx - half, y1=cy - half, x2=cx + half, y2=cy + half).clamped(
        sw, sh
    )
    return IconCandidate(
        label=target_label,
        box=bbox,
        confidence=min(1.0, chosen.score),
        is_target_match=True,
        reasoning=f"OCR taskbar '{merged}' score={chosen.score:.2f}",
        source="ocr",
        click_xy=(cx, cy),
        force_verify=True,
    )


def verify_crop_contains_label(crop: Image.Image, target_label: str) -> bool | None:
    if sys.platform != "win32":
        return None
    engine = _ocr_engine()
    if engine is None:
        return None
    result = _run_ocr(engine, crop)
    if not result:
        return False
    parsed = _parse_entries(result)
    for entry in parsed:
        if entry.score < 0.45:
            continue
        merged = merged_label_text(entry, parsed)
        if is_exact_icon_label(merged, target_label):
            return True
    return False


def detect_text_anchors(
    image: Image.Image,
    target_label: str | None = None,
    *,
    crop_top_y: int = 0,
    screen_height: int | None = None,
    icon_offset_px: int = _ICON_OFFSET_PX,
) -> list[tuple[int, int]]:
    if sys.platform != "win32":
        return []

    engine = _ocr_engine()
    if engine is None:
        return []

    result = _run_ocr(engine, image)
    if not result:
        return []

    sh = image.height
    screen_h = screen_height or sh
    parsed = _parse_entries(result)
    points: list[tuple[int, int]] = []
    for entry in parsed:
        if not _is_plausible_caption(entry, sh, crop_top_y=crop_top_y, screen_h=screen_h):
            continue
        merged = merged_label_text(entry, parsed)
        if target_label is not None and not is_exact_icon_label(merged, target_label):
            continue
        click = _icon_click_from_label_box(
            entry.box,
            icon_offset_px,
            crop_top_y=crop_top_y,
            screen_height=screen_h,
        )
        if click is not None:
            points.append(click)

    return points
