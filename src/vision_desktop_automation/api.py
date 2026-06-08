from __future__ import annotations

import json
from importlib import resources

import httpx
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from .config import Settings
from .models import Post

_FALLBACK = "fallback_posts.json"
_TIMEOUT = 5.0


def _posts_from_json(raw: object) -> list[Post]:
    if not isinstance(raw, list):
        raise ValueError("Unexpected API payload: expected a JSON array of posts.")
    return [Post.model_validate(item) for item in raw]


def _load_fallback_posts() -> list[Post]:
    path = resources.files(f"{__package__}.data") / _FALLBACK
    return _posts_from_json(json.loads(path.read_text(encoding="utf-8")))


def _fetch_from_url(url: str) -> list[Post]:
    response = httpx.get(url, timeout=_TIMEOUT, follow_redirects=True)
    response.raise_for_status()
    return _posts_from_json(response.json())


def fetch_posts(settings: Settings) -> list[Post]:
    url = settings.remote_api_url
    logger.info(f"Fetching posts from {url} …")

    @retry(
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_fixed(settings.retry_delay_seconds),
        reraise=True,
    )
    def _live() -> list[Post]:
        return _fetch_from_url(url)

    try:
        posts = _live()
        logger.success(f"Fetched {len(posts)} posts from the live API.")
    except httpx.ConnectError as exc:
        logger.warning(
            f"Live API unreachable ({type(exc).__name__}: {exc}); using bundled posts."
        )
        posts = _load_fallback_posts()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Live API unavailable ({exc!r}); using bundled posts.")
        posts = _load_fallback_posts()

    return posts[: settings.post_limit]
