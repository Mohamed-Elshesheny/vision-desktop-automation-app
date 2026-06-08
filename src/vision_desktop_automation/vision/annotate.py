from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..models import GroundingResult


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate(image: Image.Image, result: GroundingResult) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _font(20)

    for candidate in result.candidates:
        if candidate is result.chosen:
            continue
        box = candidate.box
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=(255, 170, 0), width=2)

    if result.chosen is not None:
        box = result.chosen.box
        cx, cy = result.chosen.center
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=(0, 220, 0), width=4)
        draw.line([cx - 18, cy, cx + 18, cy], fill=(255, 0, 0), width=3)
        draw.line([cx, cy - 18, cx, cy + 18], fill=(255, 0, 0), width=3)

        caption = (
            f"{result.chosen.label}  conf={result.chosen.confidence:.2f}  "
            f"({cx},{cy})  {result.method}  {result.elapsed_seconds:.2f}s"
        )
        tw = draw.textlength(caption, font=font)
        draw.rectangle([box.x1, box.y1 - 28, box.x1 + tw + 10, box.y1], fill=(0, 0, 0))
        draw.text((box.x1 + 5, box.y1 - 26), caption, fill=(255, 255, 255), font=font)

    return canvas


def save_annotated(image: Image.Image, result: GroundingResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    annotate(image, result).save(path)
    return path
