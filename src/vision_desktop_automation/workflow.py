from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from PIL import Image

from . import automation
from .api import fetch_posts
from .config import Settings, set_dpi_awareness
from .grounding import TemplateCache, get_grounder, locate_icon
from .grounding.ocr_locate import warmup_ocr
from .models import Post
from .vision.annotate import save_annotated
from .vision.screenshot import capture_desktop


@dataclass
class _PostStat:
    post_id: int
    status: str = "pending"
    elapsed_s: float = 0.0
    grounding_method: str = ""
    grounding_s: float = 0.0
    error: str = ""


def _make_capture(_settings: Settings) -> Callable[[], Image.Image]:
    def _capture() -> Image.Image:
        try:
            automation.ensure_desktop_clear()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"ensure_desktop_clear skipped: {exc!r}")
        return capture_desktop()

    return _capture


def _process_post(
    post: Post,
    settings: Settings,
    grounder,
    cache: TemplateCache,
    capture: Callable[[], Image.Image],
    failed_centers: list[tuple[int, int]],
) -> _PostStat:
    stat = _PostStat(post_id=post.id)
    post_start = time.perf_counter()

    logger.info(f"{'─' * 60}")
    logger.info(f"POST {post.id}/10  |  {post.title[:55]!r}")
    logger.info(f"{'─' * 60}")

    t0 = time.perf_counter()
    result, screenshot = locate_icon(
        grounder=grounder,
        settings=settings,
        capture=capture,
        template_cache=cache,
        blocked_centers=failed_centers,
    )
    stat.grounding_s = time.perf_counter() - t0
    stat.grounding_method = result.method

    logger.info(
        f"  grounding: found={result.found} method={result.method!r} "
        f"candidates={len(result.candidates)} theme={result.theme} "
        f"attempts={result.attempts} elapsed={stat.grounding_s:.2f}s"
    )
    if result.screen_size:
        logger.debug(f"  screen_size={result.screen_size}")

    annotation_path = settings.annotations_dir / f"post_{post.id}_grounding.png"
    try:
        save_annotated(screenshot, result, annotation_path)
        logger.debug(f"  annotation → {annotation_path.name}")
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"  annotation save skipped: {exc!r}")

    if not result.found or result.center is None:
        logger.warning(
            f"  grounding failed for post {post.id} — "
            "attempting shortcut-based launch as last resort."
        )
        launched = automation.launch_via_shortcut(settings.icon_label)
        if not launched or not automation.wait_for_window(
            settings.window_title_substring, settings.launch_timeout_seconds
        ):
            stat.status = "failed"
            stat.error = "icon not found and shortcut fallback also failed"
            stat.elapsed_s = time.perf_counter() - post_start
            logger.error(f"  all launch methods exhausted for post {post.id}; skipping.")
            return stat
        logger.info("  shortcut fallback succeeded — continuing with type/save.")
    else:
        logger.info(f"  clicking at {result.center}")
        automation.double_click_at(*result.center)

        if not automation.wait_for_window(
            settings.window_title_substring, settings.launch_timeout_seconds
        ):
            logger.warning("  purging template cache — click did not open the target window.")
            cache.invalidate(settings.icon_label)
            if result.center is not None:
                failed_centers.append(result.center)
                logger.info(
                    f"  blacklisted {result.center} "
                    f"({len(failed_centers)} bad coords recorded this run)."
                )
            stat.status = "failed"
            stat.error = "window did not open after click"
            stat.elapsed_s = time.perf_counter() - post_start
            logger.error(f"  {settings.window_title_substring} did not launch; skipping post.")
            return stat

    text = post.to_notepad_text()
    t0 = time.perf_counter()
    if not automation.type_text(
        text,
        settings.typing_interval_seconds,
        window_substring=settings.window_title_substring,
    ):
        stat.status = "failed"
        stat.error = "typing aborted (window lost focus)"
        stat.elapsed_s = time.perf_counter() - post_start
        logger.error(f"  typing aborted for post {post.id}; skipping.")
        automation.close_window(settings.window_title_substring)
        return stat
    logger.debug(f"  typed {len(text)} chars in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    saved = automation.save_as(
        settings.save_dir / post.filename,
        settings.window_title_substring,
        settings.typing_interval_seconds,
    )
    logger.debug(f"  save_as took {time.perf_counter() - t0:.2f}s → saved={saved}")

    automation.close_window(settings.window_title_substring)

    stat.status = "ok" if saved else "failed"
    if not saved:
        stat.error = "save_as returned False"
    stat.elapsed_s = time.perf_counter() - post_start
    logger.info(f"  post {post.id} {stat.status.upper()} in {stat.elapsed_s:.1f}s")
    return stat


def _run_pass(
    posts: list[Post],
    settings: Settings,
    grounder,
    cache: TemplateCache,
    capture: Callable[[], Image.Image],
    failed_centers: list[tuple[int, int]],
    pass_label: str,
) -> tuple[int, list[Post], list[_PostStat]]:
    succeeded = 0
    still_failed: list[Post] = []
    stats: list[_PostStat] = []
    interrupted = False

    for post in posts:
        try:
            stat = _process_post(post, settings, grounder, cache, capture, failed_centers)
            stats.append(stat)
            if stat.status == "ok":
                succeeded += 1
            else:
                still_failed.append(post)
        except KeyboardInterrupt:
            logger.warning(f"Interrupted during {pass_label}; stopping.")
            interrupted = True
            break
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Unhandled error on post {post.id}: {exc!r}")
            err_stat = _PostStat(post_id=post.id, status="failed", error=str(exc))
            stats.append(err_stat)
            still_failed.append(post)

    if interrupted:
        raise KeyboardInterrupt

    return succeeded, still_failed, stats


def _print_run_summary(
    all_stats: list[_PostStat], total_s: float, save_dir: Path
) -> None:
    ok = [s for s in all_stats if s.status == "ok"]
    failed = [s for s in all_stats if s.status != "ok"]

    logger.info("═" * 60)
    logger.info(f"  RUN SUMMARY  —  {len(ok)}/{len(all_stats)} posts saved")
    logger.info("═" * 60)
    logger.info(f"  {'Post':<6}  {'Status':<8}  {'Total':>7}  {'Ground':>7}  {'Method'}")
    logger.info(f"  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*20}")
    for s in sorted(all_stats, key=lambda x: x.post_id):
        status_tag = "✓ OK" if s.status == "ok" else "✗ FAIL"
        logger.info(
            f"  {s.post_id:<6}  {status_tag:<8}  "
            f"{s.elapsed_s:>6.1f}s  {s.grounding_s:>6.1f}s  "
            f"{s.grounding_method or '—'}"
        )
    logger.info("─" * 60)
    avg_ground = (
        sum(s.grounding_s for s in all_stats) / len(all_stats) if all_stats else 0
    )
    logger.info(f"  Total wall time : {total_s:.1f}s")
    logger.info(f"  Avg grounding   : {avg_ground:.1f}s/post")
    logger.info(f"  Output dir      : {save_dir}")
    if failed:
        logger.warning(f"  Failed posts    : {[s.post_id for s in failed]}")
        for s in failed:
            if s.error:
                logger.warning(f"    post {s.post_id}: {s.error}")
    logger.info("═" * 60)


def run_workflow(settings: Settings) -> int:
    set_dpi_awareness()
    settings.ensure_dirs()
    warmup_ocr()

    posts = fetch_posts(settings)
    logger.info(f"Processing {len(posts)} posts.")
    logger.debug(
        f"Settings: max_retries={settings.max_retries} "
        f"retry_delay={settings.retry_delay_seconds}s "
        f"launch_timeout={settings.launch_timeout_seconds}s "
        f"typing_interval={settings.typing_interval_seconds}s"
    )

    grounder = get_grounder(settings)
    cache = TemplateCache(settings)
    capture = _make_capture(settings)
    failed_centers: list[tuple[int, int]] = []
    all_stats: list[_PostStat] = []
    run_start = time.perf_counter()
    succeeded = 0

    try:
        succeeded, failed_posts, stats = _run_pass(
            posts, settings, grounder, cache, capture, failed_centers, "first pass"
        )
        all_stats.extend(stats)

        if failed_posts:
            logger.info(
                f"Retry pass: {len(failed_posts)} post(s) failed; "
                f"blacklist now has {len(failed_centers)} coord(s). Retrying..."
            )
            retry_succeeded, _, retry_stats = _run_pass(
                failed_posts, settings, grounder, cache, capture, failed_centers, "retry pass"
            )
            retried_ids = {s.post_id for s in retry_stats}
            all_stats = [s for s in all_stats if s.post_id not in retried_ids]
            all_stats.extend(retry_stats)
            succeeded += retry_succeeded

    except KeyboardInterrupt:
        logger.warning("Run interrupted by user.")

    total_s = time.perf_counter() - run_start
    _print_run_summary(all_stats, total_s, settings.save_dir)
    logger.info(f"Done: {succeeded}/{len(posts)} posts saved to {settings.save_dir}.")
    return 0 if succeeded == len(posts) else 1


def run_demo(settings: Settings, tag: str) -> int:
    set_dpi_awareness()
    settings.ensure_dirs()
    warmup_ocr()

    grounder = get_grounder(settings)
    cache = TemplateCache(settings)
    capture = _make_capture(settings)

    result, screenshot = locate_icon(
        grounder=grounder, settings=settings, capture=capture, template_cache=cache
    )
    out = settings.deliverables_dir / f"grounding_{tag}.png"
    save_annotated(screenshot, result, out)

    if result.found and result.center is not None:
        logger.success(
            f"[{tag}] {settings.icon_label} at {result.center} "
            f"({result.method}, {result.elapsed_seconds:.2f}s) -> {out}"
        )
        return 0
    logger.error(f"[{tag}] icon not found -> {out}")
    return 1
