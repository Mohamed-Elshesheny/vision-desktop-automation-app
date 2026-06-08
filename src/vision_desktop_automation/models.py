from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Post(BaseModel):
    id: int
    title: str
    body: str
    userId: int | None = None

    def to_notepad_text(self) -> str:
        return f"Title: {self.title}\n\n{self.body}"

    @property
    def filename(self) -> str:
        return f"post_{self.id}.txt"


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

    @field_validator("x2")
    @classmethod
    def _x_order(cls, v: int, info) -> int:  # type: ignore[no-untyped-def]
        x1 = info.data.get("x1", 0)
        return max(v, x1 + 1)

    @field_validator("y2")
    @classmethod
    def _y_order(cls, v: int, info) -> int:  # type: ignore[no-untyped-def]
        y1 = info.data.get("y1", 0)
        return max(v, y1 + 1)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)

    def clamped(self, max_w: int, max_h: int) -> BoundingBox:
        return BoundingBox(
            x1=max(0, min(self.x1, max_w - 1)),
            y1=max(0, min(self.y1, max_h - 1)),
            x2=max(1, min(self.x2, max_w)),
            y2=max(1, min(self.y2, max_h)),
        )


class IconCandidate(BaseModel):
    label: str
    box: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    is_target_match: bool = True
    reasoning: str = ""
    source: str = "vlm"
    click_xy: tuple[int, int] | None = None
    force_verify: bool = False

    @property
    def center(self) -> tuple[int, int]:
        return self.click_xy if self.click_xy is not None else self.box.center


class GroundingResult(BaseModel):
    found: bool
    candidates: list[IconCandidate] = Field(default_factory=list)
    chosen: IconCandidate | None = None
    attempts: int = 0
    elapsed_seconds: float = 0.0
    method: str = ""
    theme: str = "unknown"
    screen_size: tuple[int, int] | None = None
    error: str | None = None

    @property
    def center(self) -> tuple[int, int] | None:
        return self.chosen.center if self.chosen else None
