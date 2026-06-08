from __future__ import annotations

from loguru import logger
from PIL import Image

from ...config import Settings
from ...vision.screenshot import pil_to_png_base64
from ..base import VerifyResult, VisionGrounder
from ..prompts import (
    GROUND_SYSTEM,
    coarse_grid_prompt,
    point_grid_prompt,
    verify_prompt,
)


class OpenAIGrounder(VisionGrounder):
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        from openai import OpenAI

        self._client = OpenAI(api_key=settings.require_api_key(), timeout=60.0)
        self._model = settings.openai_model
        self._grid_model: str = getattr(settings, "grid_model", "gpt-4o") or self._model

    def _chat(self, image: Image.Image, prompt: str, detail: str = "high", model: str | None = None) -> str:
        import time as _time
        use_model = model or self._model
        b64 = pil_to_png_base64(image)
        t0 = _time.perf_counter()
        logger.debug(
            f"  [openai] _chat: model={use_model} detail={detail} "
            f"img={image.width}×{image.height} "
            f"prompt_chars={len(prompt)}"
        )
        response = self._client.chat.completions.create(
            model=use_model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": GROUND_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                                "detail": detail,
                            },
                        },
                    ],
                },
            ],
        )
        elapsed = _time.perf_counter() - t0
        content = response.choices[0].message.content or ""
        usage = response.usage
        if usage:
            logger.debug(
                f"  [openai] response in {elapsed:.2f}s | "
                f"tokens: prompt={usage.prompt_tokens} "
                f"completion={usage.completion_tokens} "
                f"total={usage.total_tokens}"
            )
        else:
            logger.debug(f"  [openai] response in {elapsed:.2f}s (no usage info)")
        logger.debug(f"  [openai] raw response: {content[:300]}")
        return content

    def verify(self, image: Image.Image, icon_label: str) -> VerifyResult:
        try:
            text = self._chat(image, verify_prompt(icon_label))
            return VerifyResult.model_validate_json(text.strip().strip("`"))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[openai] verify call failed: {exc!r}")
            return VerifyResult(match=False, reasoning=str(exc))

    def ground_grid_cell(
        self,
        image: Image.Image,
        icon_label: str,
        excluded_cells: list[str] | None = None,
    ) -> tuple[str | None, str | None, float]:
        try:
            import json as _json

            text = self._chat(
                image,
                coarse_grid_prompt(icon_label, excluded_cells=excluded_cells),
                detail="auto",
                model=self._grid_model,
            )
            cleaned = text.strip().strip("`").removeprefix("json").strip()
            data = _json.loads(cleaned)

            def _parse(key: str) -> str | None:
                raw = str(data.get(key, "NONE")).upper().strip()
                return None if raw in ("NONE", "", "NULL") else raw

            primary = _parse("cell")
            backup = _parse("backup_cell")
            conf = float(data.get("confidence", 0.5))
            logger.debug(
                f"  [openai] grid cell: primary={primary} backup={backup} conf={conf:.2f}"
            )
            return primary, backup, conf
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[openai] grid cell call failed: {exc!r}")
            return None, None, 0.0

    def ground_crosshair_point(self, image: Image.Image, icon_label: str) -> int:
        try:
            import json as _json

            text = self._chat(
                image, point_grid_prompt(icon_label), model=self._grid_model, detail="high"
            )
            cleaned = text.strip().strip("`").removeprefix("json").strip()
            data = _json.loads(cleaned)
            return int(data.get("point_id", -1))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[openai] crosshair point call failed: {exc!r}")
            return -1
