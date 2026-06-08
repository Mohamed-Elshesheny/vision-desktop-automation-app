from __future__ import annotations

import string
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path as _Path

from loguru import logger
from PIL import Image

from ..config import Settings
from ..models import BoundingBox, GroundingResult, IconCandidate
from ..vision.preprocess import (
    GRID_COLS,
    GRID_ROWS,
    cell_bounds,
    detect_theme,
    draw_anchor_marks,
    draw_grid_overlay,
    resize_for_vlm,
)
from .base import VisionGrounder
from .ocr_locate import detect_text_anchors, locate_by_label_ocr, verify_crop_contains_label
from .template_cache import TemplateCache

CaptureFn = Callable[[], Image.Image]
_BLACKLIST_RADIUS_PX = 35
_TEMPLATE_VERIFY_SKIP = 0.85
_MAX_GRID_ROUNDS = 4


@dataclass
class _SpatialState:
    blocked_centers: list[tuple[int, int]] = field(default_factory=list)
    excluded_cells: set[str] = field(default_factory=set)

    def is_blocked(self, center: tuple[int, int]) -> bool:
        cx, cy = center
        return any(
            ((cx - bx) ** 2 + (cy - by) ** 2) ** 0.5 < _BLACKLIST_RADIUS_PX
            for bx, by in self.blocked_centers
        )

    def record_failure(self, center: tuple[int, int], sw: int, sh: int) -> None:
        cell = _pixel_to_cell(center[0], center[1], sw, sh)
        if cell not in self.excluded_cells:
            self.excluded_cells.add(cell)
            logger.info(
                f"  excluded region: coord {center} → cell {cell} "
                f"({len(self.excluded_cells)} cells, {len(self.blocked_centers)} coords)"
            )


def _pixel_to_cell(x: int, y: int, sw: int, sh: int) -> str:
    col = min(int(x / sw * GRID_COLS), GRID_COLS - 1)
    row = min(int(y / sh * GRID_ROWS), GRID_ROWS - 1)
    return f"{string.ascii_uppercase[row]}{col + 1}"


def _resolve_anchor_point_id(raw_id: int, count: int) -> int | None:
    if raw_id < 0:
        return None
    if raw_id < count:
        return raw_id
    if 1 <= raw_id <= count:
        adjusted = raw_id - 1
        logger.info(
            f"  grid stage-2: normalized point_id {raw_id} → {adjusted} (1-based → 0-based)"
        )
        return adjusted
    return None


def _verify(
    grounder: VisionGrounder,
    screenshot: Image.Image,
    candidate: IconCandidate,
    target_label: str,
    settings: Settings,
) -> bool:
    if not settings.verify_enabled:
        return True
    sw, sh = screenshot.size
    cx, cy = candidate.center
    margin = 80
    crop = screenshot.crop(
        (max(0, cx - margin), max(0, cy - margin), min(sw, cx + margin), min(sh, cy + margin))
    )
    local = verify_crop_contains_label(crop, target_label)
    if local is False:
        logger.info(
            f"  verify: local OCR found no exact '{target_label}' in crop — reject."
        )
        return False
    result = grounder.verify(crop, target_label)
    logger.info(f"  verify: match={result.match} ({result.reasoning[:60]})")
    return result.match


def _template_fast_path(
    grounder: VisionGrounder,
    screenshot: Image.Image,
    target_label: str,
    theme: str,
    cache: TemplateCache,
    settings: Settings,
    spatial: _SpatialState,
) -> IconCandidate | None:
    match = cache.match(screenshot, target_label, theme)
    if match is None:
        return None

    score, box = match
    candidate = IconCandidate(
        label=target_label,
        box=box,
        confidence=min(1.0, score),
        is_target_match=True,
        reasoning=f"template match score={score:.2f}",
        source="template",
    )

    if spatial.is_blocked(candidate.center):
        logger.warning(f"  [template] {candidate.center} excluded — skip.")
        return None

    if score >= _TEMPLATE_VERIFY_SKIP:
        logger.info(f"  template conf={score:.2f} — skipping verify.")
        return candidate

    sw, sh = screenshot.size
    if settings.verify_enabled and not _verify(grounder, screenshot, candidate, target_label, settings):
        cache.invalidate(target_label)
        spatial.record_failure(candidate.center, sw, sh)
        logger.warning(
            f"  template verification failed at {candidate.center} — cache cleared."
        )
        return None
    return candidate


def _cell_to_candidate(
    grounder: VisionGrounder,
    screenshot: Image.Image,
    cell: str,
    target_label: str,
    sw: int,
    sh: int,
    debug_dir: _Path | None,
    grid_conf: float,
    spatial: _SpatialState,
    settings: Settings,
) -> IconCandidate | None:
    bounds = cell_bounds(cell, sw, sh)
    if bounds is None:
        logger.info(f"  grid: invalid cell '{cell}'")
        return None

    x0, y0, x1, y1 = bounds
    cell_w, cell_h = x1 - x0, y1 - y0
    mx, my = int(cell_w * 0.5), int(cell_h * 0.5)
    crop_rect = (
        max(0, x0 - mx),
        max(0, y0 - my),
        min(sw, x1 + mx),
        min(sh, y1 + my),
    )
    crop = screenshot.crop(crop_rect)
    logger.info(f"  grid stage-2 [{cell}]: crop {crop_rect}")

    icon_offset = settings.scaled_icon_offset(sh)
    label_points = detect_text_anchors(
        crop,
        target_label,
        crop_top_y=crop_rect[1],
        screen_height=sh,
        icon_offset_px=icon_offset,
    )
    if len(label_points) == 1:
        points = label_points
        resolved_id = 0
        logger.info(
            f"  grid stage-2 [{cell}]: single '{target_label}' anchor — skip crosshair VLM."
        )
    else:
        points = label_points
        if not points:
            points = detect_text_anchors(
                crop,
                crop_top_y=crop_rect[1],
                screen_height=sh,
                icon_offset_px=icon_offset,
            )
        if not points:
            logger.info(f"  grid stage-2 [{cell}]: no OCR anchors — skip cell.")
            return None

        marked = draw_anchor_marks(crop, points)

        try:
            point_id = grounder.ground_crosshair_point(marked, target_label)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"  grid stage-2 anchor call failed: {exc!r}")
            return None

        resolved_id = _resolve_anchor_point_id(point_id, len(points))
        if resolved_id is None:
            logger.info(f"  grid stage-2 [{cell}]: no valid anchor (point_id={point_id})")
            return None

    if debug_dir is not None:
        try:
            draw_anchor_marks(crop, points).save(_Path(debug_dir) / f"anchors_{cell}.png")
        except Exception:  # noqa: BLE001
            pass

    px, py = points[resolved_id]
    screen_x = crop_rect[0] + px
    screen_y = crop_rect[1] + py
    if spatial.is_blocked((screen_x, screen_y)):
        logger.info(f"  grid stage-2 [{cell}]: anchor lands on excluded coord ({screen_x}, {screen_y})")
        return None

    icon_half = max(32, int(min(cell_w, cell_h) * 0.15))
    box = BoundingBox(
        x1=screen_x - icon_half,
        y1=screen_y - icon_half,
        x2=screen_x + icon_half,
        y2=screen_y + icon_half,
    ).clamped(sw, sh)

    logger.info(f"  grid stage-2 [{cell}]: anchor {resolved_id} → ({screen_x}, {screen_y})")
    return IconCandidate(
        label=target_label,
        box=box,
        confidence=min(0.99, max(0.70, grid_conf)),
        is_target_match=True,
        reasoning=f"grid cell {cell}, anchor {resolved_id}",
        source="grid",
        click_xy=(screen_x, screen_y),
    )


def _vlm_grid_locate(
    grounder: VisionGrounder,
    screenshot: Image.Image,
    target_label: str,
    spatial: _SpatialState,
    settings: Settings,
    debug_dir: _Path | None,
) -> IconCandidate | None:
    sw, sh = screenshot.size
    grid_img = draw_grid_overlay(screenshot)
    if debug_dir is not None:
        try:
            _Path(debug_dir).mkdir(parents=True, exist_ok=True)
            grid_img.save(_Path(debug_dir) / "grid_stage1.png")
        except Exception:  # noqa: BLE001
            pass

    grid_small = resize_for_vlm(grid_img)

    for grid_round in range(1, _MAX_GRID_ROUNDS + 1):
        excluded = sorted(spatial.excluded_cells)
        if excluded:
            logger.info(f"  grid stage-1 (round {grid_round}): excluding {excluded}")

        try:
            primary, backup, grid_conf = grounder.ground_grid_cell(
                grid_small, target_label, excluded_cells=excluded or None
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"  grid stage-1 failed: {exc!r}")
            return None

        if not primary:
            logger.info("  grid stage-1: model returned NONE.")
            return None

        logger.info(
            f"  grid stage-1: primary={primary} backup={backup} conf={grid_conf:.2f}"
        )
        cells = [
            c
            for c in (primary, backup)
            if c and c.upper() not in spatial.excluded_cells
        ]
        if not cells:
            logger.info("  grid stage-1: all returned cells already excluded — re-asking.")
            continue

        for cell in cells:
            candidate = _cell_to_candidate(
                grounder,
                screenshot,
                cell,
                target_label,
                sw,
                sh,
                debug_dir,
                grid_conf,
                spatial,
                settings,
            )
            if candidate is None:
                spatial.excluded_cells.add(cell.upper())
                continue
            if spatial.is_blocked(candidate.center):
                spatial.record_failure(candidate.center, sw, sh)
                continue
            if _verify(grounder, screenshot, candidate, target_label, settings):
                return candidate
            spatial.record_failure(candidate.center, sw, sh)
            logger.warning(f"  grid verification failed at {candidate.center} — cell {cell} excluded.")

    return None


def _attempt_once(
    grounder: VisionGrounder,
    settings: Settings,
    screenshot: Image.Image,
    target_label: str,
    theme: str,
    cache: TemplateCache,
    spatial: _SpatialState,
) -> tuple[IconCandidate | None, list[IconCandidate], list[str]]:
    sw, sh = screenshot.size
    methods: list[str] = []
    candidates: list[IconCandidate] = []

    if settings.template_fast_path:
        candidate = _template_fast_path(
            grounder, screenshot, target_label, theme, cache, settings, spatial
        )
        if candidate is not None:
            return candidate, [candidate], ["template"]

    if settings.ocr_fast_path:
        methods.append("ocr")
        candidate = locate_by_label_ocr(
            screenshot,
            target_label,
            icon_offset_px=settings.scaled_icon_offset(sh),
            scan_right_column=settings.ocr_scan_right_column,
        )
        if candidate is not None:
            candidates.append(candidate)
            if spatial.is_blocked(candidate.center):
                logger.warning(f"  [ocr] {candidate.center} excluded — skip.")
            elif settings.verify_enabled or candidate.force_verify:
                if _verify(grounder, screenshot, candidate, target_label, settings):
                    return candidate, [candidate], ["ocr"]
                cache.invalidate(target_label)
                spatial.record_failure(candidate.center, sw, sh)
                logger.warning(
                    f"  OCR verification failed at {candidate.center} — cache cleared."
                )
            else:
                return candidate, [candidate], ["ocr"]

    methods.append(f"{grounder.name}:grid")
    candidate = _vlm_grid_locate(
        grounder,
        screenshot,
        target_label,
        spatial,
        settings,
        settings.screenshots_dir,
    )
    if candidate is not None:
        return candidate, [candidate], methods

    return None, candidates, methods


def locate_icon(
    *,
    grounder: VisionGrounder,
    settings: Settings,
    capture: CaptureFn,
    icon_label: str | None = None,
    template_cache: TemplateCache | None = None,
    blocked_centers: list[tuple[int, int]] | None = None,
) -> tuple[GroundingResult, Image.Image]:
    target_label = icon_label or settings.icon_label
    cache = template_cache or TemplateCache(settings)
    spatial = _SpatialState(blocked_centers=list(blocked_centers or []))
    last_screenshot: Image.Image | None = None
    last_candidates: list[IconCandidate] = []
    start = time.perf_counter()

    for attempt in range(1, settings.max_retries + 1):
        screenshot = capture()
        last_screenshot = screenshot
        sw, sh = screenshot.size
        theme = detect_theme(screenshot)
        logger.info(
            f"Grounding attempt {attempt}/{settings.max_retries} "
            f"for '{target_label}' ({sw}x{sh}, {theme} theme)"
            + (
                f" [excluded: {len(spatial.excluded_cells)} cells, "
                f"{len(spatial.blocked_centers)} coords]"
                if spatial.excluded_cells or spatial.blocked_centers
                else ""
            )
        )

        chosen, candidates, methods = _attempt_once(
            grounder, settings, screenshot, target_label, theme, cache, spatial
        )
        if candidates:
            last_candidates = candidates

        if chosen is not None:
            cache.update(screenshot, chosen.box, target_label, theme)
            elapsed = time.perf_counter() - start
            logger.success(
                f"  grounded '{target_label}' at {chosen.center} "
                f"in {elapsed:.2f}s via {'+'.join(methods)}"
            )
            return (
                GroundingResult(
                    found=True,
                    candidates=candidates,
                    chosen=chosen,
                    attempts=attempt,
                    elapsed_seconds=elapsed,
                    method="+".join(methods),
                    theme=theme,
                    screen_size=(sw, sh),
                ),
                screenshot,
            )

        time.sleep(settings.retry_delay_seconds)

    elapsed = time.perf_counter() - start
    logger.error(f"Failed to ground '{target_label}' after {settings.max_retries} attempts.")
    screen_size = last_screenshot.size if last_screenshot else None
    return (
        GroundingResult(
            found=False,
            candidates=last_candidates,
            attempts=settings.max_retries,
            elapsed_seconds=elapsed,
            method="exhausted",
            screen_size=screen_size,
            error=f"icon '{target_label}' not found",
        ),
        last_screenshot or Image.new("RGB", (1, 1)),
    )
