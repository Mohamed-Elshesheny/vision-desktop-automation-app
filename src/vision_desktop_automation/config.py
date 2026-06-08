from __future__ import annotations

import contextlib
import ctypes
import sys
from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"

    icon_label: str = "Notepad"
    window_title_substring: str = "Notepad"

    remote_api_url: str = "https://jsonplaceholder.typicode.com/posts"
    post_limit: int = 10

    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    launch_timeout_seconds: float = 5.0
    typing_interval_seconds: float = 0.01

    verify_enabled: bool = True
    template_fast_path: bool = True
    ocr_fast_path: bool = True
    ocr_scan_right_column: bool = False
    template_match_threshold: float = 0.45
    icon_offset_px: int = 32

    grid_model: str = "gpt-4o"
    save_dir: Path = Field(default_factory=lambda: Path.home() / "Desktop" / "tjm-project")
    output_dir: Path = Field(default_factory=lambda: Path.cwd() / "output")

    @property
    def screenshots_dir(self) -> Path:
        return self.output_dir / "screenshots"

    @property
    def annotations_dir(self) -> Path:
        return self.output_dir / "annotations"

    @property
    def deliverables_dir(self) -> Path:
        return Path.cwd() / "docs" / "deliverables"

    @property
    def templates_dir(self) -> Path:
        return self.output_dir / "templates"

    def ensure_dirs(self) -> None:
        for path in (
            self.save_dir,
            self.screenshots_dir,
            self.annotations_dir,
            self.templates_dir,
            self.deliverables_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def scaled_icon_offset(self, screen_height: int, *, baseline: int = 1080) -> int:
        return max(20, int(self.icon_offset_px * screen_height / baseline))

    def require_api_key(self) -> str:
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env (copy .env.example)."
            )
        return self.openai_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def set_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    with contextlib.suppress(AttributeError, OSError):
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # type: ignore[attr-defined]
        return
    with contextlib.suppress(AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
