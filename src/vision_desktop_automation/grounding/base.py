from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image
from pydantic import BaseModel

from ..config import Settings


class VerifyResult(BaseModel):
    match: bool = False
    reasoning: str = ""


def normalize_icon_label(label: str) -> str:
    return label.strip().lower()


def is_exact_icon_label(candidate_label: str, target_label: str) -> bool:
    return normalize_icon_label(candidate_label) == normalize_icon_label(target_label)


class VisionGrounder(ABC):
    name: str = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def verify(self, image: Image.Image, icon_label: str) -> VerifyResult:
        pass

    def ground_grid_cell(
        self,
        image: Image.Image,
        icon_label: str,
        excluded_cells: list[str] | None = None,
    ) -> tuple[str | None, str | None, float]:
        raise NotImplementedError

    def ground_crosshair_point(self, image: Image.Image, icon_label: str) -> int:
        raise NotImplementedError


def get_grounder(settings: Settings) -> VisionGrounder:
    from .providers.openai import OpenAIGrounder

    return OpenAIGrounder(settings)
