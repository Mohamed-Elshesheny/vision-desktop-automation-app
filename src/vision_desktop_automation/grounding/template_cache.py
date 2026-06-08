from __future__ import annotations

import os

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from ..config import Settings
from ..models import BoundingBox
from ..vision.preprocess import to_edge_image

_MIN_GOOD_MATCHES = 4
_MAX_MATCH_DISTANCE = 50
_MIN_KEYPOINTS = 6
_SCALES = np.linspace(0.6, 1.5, 10)
_ORB_SCALE = 0.5


def _to_gray_array(image: Image.Image) -> np.ndarray:
    arr = np.array(image.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def _resize_half(image: Image.Image) -> Image.Image:
    w, h = image.size
    return image.resize((max(8, w // 2), max(8, h // 2)), Image.BILINEAR)


class TemplateCache:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._orb = cv2.ORB_create(nfeatures=500)
        self._kp_cache: dict[str, tuple] = {}

    def _path(self, label: str, theme: str):
        safe = "".join(c for c in label if c.isalnum() or c in "-_").lower()
        return self.settings.templates_dir / f"{safe}_{theme}.png"

    def update(self, screenshot: Image.Image, box: BoundingBox, label: str, theme: str) -> None:
        try:
            self.settings.templates_dir.mkdir(parents=True, exist_ok=True)
            crop = screenshot.crop((box.x1, box.y1, box.x2, box.y2))
            crop.save(self._path(label, theme))
            logger.debug(f"Template cache updated: {self._path(label, theme)}")
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Template cache update skipped: {exc!r}")

    def invalidate(self, label: str) -> None:
        for theme in ("dark", "light"):
            path = self._path(label, theme)
            self._kp_cache.pop(str(path), None)
            if path.exists():
                try:
                    path.unlink()
                    logger.info(f"  template cache invalidated: {path.name}")
                except OSError as exc:
                    logger.debug(f"  could not delete template {path.name}: {exc!r}")

    def _template_kp(
        self, path, template: Image.Image
    ) -> tuple[list, np.ndarray | None]:
        key = str(path)
        try:
            mtime = os.path.getmtime(key)
        except OSError:
            mtime = -1.0
        cached = self._kp_cache.get(key)
        if cached is not None and cached[2] == mtime:
            return cached[0], cached[1]

        small = _resize_half(template)
        gray = _to_gray_array(small)
        kp, des = self._orb.detectAndCompute(gray, None)
        self._kp_cache[key] = (kp, des, mtime)
        return kp, des

    def _match_orb(
        self, path, template: Image.Image, screenshot: Image.Image
    ) -> tuple[float, BoundingBox] | None:
        kp1, des1 = self._template_kp(path, template)
        sw, sh = screenshot.size

        small_screen = _resize_half(screenshot)
        scr_gray = _to_gray_array(small_screen)
        kp2, des2 = self._orb.detectAndCompute(scr_gray, None)

        if des1 is None or des2 is None or len(kp1) < _MIN_KEYPOINTS:
            return None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        raw_matches = bf.match(des1, des2)
        good = [m for m in raw_matches if m.distance < _MAX_MATCH_DISTANCE]

        logger.debug(
            f"  ORB: {len(kp1)} tpl kp, {len(kp2)} scr kp, "
            f"{len(raw_matches)} raw matches, {len(good)} good."
        )

        if len(good) < _MIN_GOOD_MATCHES:
            return None

        confidence = min(1.0, len(good) / max(1, len(kp1)))
        if confidence < self.settings.template_match_threshold:
            return None

        inv_scale = 1.0 / _ORB_SCALE
        pts = np.float32([kp2[m.trainIdx].pt for m in good]) * inv_scale
        x1, y1 = pts.min(axis=0).astype(int)
        x2, y2 = pts.max(axis=0).astype(int)

        pad_x = max(8, template.width // 2)
        pad_y = max(8, template.height // 2)
        box = BoundingBox(
            x1=max(0, x1 - pad_x),
            y1=max(0, y1 - pad_y),
            x2=min(sw, x2 + pad_x),
            y2=min(sh, y2 + pad_y),
        ).clamped(sw, sh)

        return confidence, box

    def _match_canny(
        self, template: Image.Image, screenshot: Image.Image
    ) -> tuple[float, BoundingBox] | None:
        screen_edges = to_edge_image(screenshot)
        template_edges = to_edge_image(template)
        screen_h, screen_w = screen_edges.shape

        best_score = -1.0
        best_box: BoundingBox | None = None

        for scale in _SCALES:
            tw = int(template_edges.shape[1] * scale)
            th = int(template_edges.shape[0] * scale)
            if tw < 8 or th < 8 or tw >= screen_w or th >= screen_h:
                continue
            resized = cv2.resize(template_edges, (tw, th))
            result = cv2.matchTemplate(screen_edges, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = float(max_val)
                x, y = max_loc
                best_box = BoundingBox(
                    x1=x, y1=y, x2=x + tw, y2=y + th
                ).clamped(screen_w, screen_h)

        if best_box is None or best_score < self.settings.template_match_threshold:
            return None
        return best_score, best_box

    def match(
        self, screenshot: Image.Image, label: str, theme: str
    ) -> tuple[float, BoundingBox] | None:
        path = self._path(label, theme)
        if not path.exists():
            return None

        template = Image.open(path).convert("RGB")

        result = self._match_orb(path, template, screenshot)
        if result is not None:
            logger.debug(f"  template fast-path: ORB match (conf={result[0]:.2f})")
            return result

        logger.debug("  ORB keypoints insufficient — falling back to Canny+matchTemplate.")
        result = self._match_canny(template, screenshot)
        if result is not None:
            logger.debug(f"  template fast-path: Canny match (score={result[0]:.2f})")
        return result
