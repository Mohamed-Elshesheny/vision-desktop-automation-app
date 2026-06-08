from __future__ import annotations

import os
import time
from pathlib import Path

import pyautogui
import pygetwindow
from loguru import logger

from .grounding.base import is_exact_icon_label

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05


def active_window_title() -> str:
    try:
        win = pygetwindow.getActiveWindow()
        return win.title if win else ""
    except Exception:  # noqa: BLE001
        return ""


def windows_with_title(substring: str):
    try:
        return pygetwindow.getWindowsWithTitle(substring)
    except Exception:  # noqa: BLE001
        return []


def is_window_focused(substring: str) -> bool:
    title = active_window_title().lower()
    sub = substring.lower()
    return title == sub or title.endswith(f" {sub}")


def wait_for_window(substring: str, timeout: float) -> bool:
    t0 = time.monotonic()
    deadline = t0 + timeout
    logger.debug(f"  wait_for_window: watching for '{substring}' (timeout={timeout}s)")
    while time.monotonic() < deadline:
        matches = windows_with_title(substring)
        if matches:
            try:
                matches[0].activate()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.25)
            if is_window_focused(substring):
                elapsed = time.monotonic() - t0
                logger.success(f"  window '{substring}' focused in {elapsed:.2f}s.")
                return True
        time.sleep(0.25)
    logger.error(
        f"  timed out waiting for window '{substring}' after {timeout}s."
    )
    return False


def launch_via_shortcut(icon_label: str) -> bool:
    desktop_dirs = [
        Path(os.path.expanduser("~/Desktop")),
        Path(os.environ.get("PUBLIC", "C:/Users/Public")) / "Desktop",
    ]
    for desktop in desktop_dirs:
        if not desktop.exists():
            continue
        for ext in ("*.lnk", "*.exe"):
            for candidate in desktop.glob(ext):
                if is_exact_icon_label(candidate.stem, icon_label):
                    try:
                        logger.warning(
                            f"  [fallback] VLM unavailable — launching via shortcut: {candidate}"
                        )
                        os.startfile(str(candidate))
                        return True
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(f"  [fallback] os.startfile failed: {exc!r}")

    logger.error(
        f"  [fallback] no shortcut found for '{icon_label}' on Desktop — "
        "cannot launch without VLM."
    )
    return False


def ensure_desktop_clear() -> None:
    logger.info("Minimizing windows to reveal the desktop.")
    pyautogui.hotkey("winleft", "m")
    time.sleep(0.3)
    width, height = pyautogui.size()
    pyautogui.moveTo(width - 5, height - 5)


def double_click_at(x: int, y: int) -> None:
    logger.info(f"Double-clicking at ({x}, {y}).")
    pyautogui.moveTo(x, y, duration=0.15)
    pyautogui.doubleClick(x, y)


def clear_document() -> None:
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.15)
    pyautogui.press("delete")
    time.sleep(0.1)


def _paste_text(text: str) -> bool:
    try:
        import pyperclip
        pyperclip.copy(text)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"  clipboard copy failed ({exc!r}); will fall back to write().")
        return False
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.1)
    return True


def type_text(text: str, interval: float, window_substring: str = "") -> bool:
    if window_substring and not is_window_focused(window_substring):
        logger.error(
            f"  type_text: '{window_substring}' is not focused; aborting."
        )
        return False
    clear_document()
    t0 = time.perf_counter()
    logger.info(f"  pasting {len(text)} chars via clipboard.")
    if not _paste_text(text):
        logger.debug(f"  clipboard unavailable — falling back to write() at {interval*1000:.0f}ms/char.")
        pyautogui.write(text, interval=interval)
    elapsed = time.perf_counter() - t0
    logger.debug(f"  type_text done in {elapsed:.3f}s ({len(text)/max(elapsed,0.001):.0f} chars/s)")
    return True


def save_as(path: Path, window_substring: str, typing_interval: float) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            path.unlink()
            logger.info(f"  removed existing file {path.name}.")
        except OSError:
            pass

    if not is_window_focused(window_substring):
        logger.error("  not focused on target window; aborting save.")
        return False

    for win in windows_with_title(window_substring):
        try:
            win.activate()
            time.sleep(0.1)
            break
        except Exception:  # noqa: BLE001
            pass

    t0_save = time.perf_counter()
    logger.debug(f"  save_as: opening dialog for {path}")
    pyautogui.hotkey("ctrl", "shift", "s")
    time.sleep(0.7)

    title = active_window_title().lower()
    if "save" not in title:
        time.sleep(0.5)
        title = active_window_title().lower()
        if "save" not in title:
            logger.error(f"  Save As dialog did not appear (active: '{title}').")
            pyautogui.press("escape")
            return False

    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.1)
    if not _paste_text(str(path)):
        pyautogui.write(str(path), interval=typing_interval)
    pyautogui.press("enter")
    time.sleep(0.5)

    confirm_title = active_window_title().lower()
    if "confirm" in confirm_title or "replace" in confirm_title or "overwrite" in confirm_title:
        pyautogui.press("left")
        pyautogui.press("enter")
        time.sleep(0.2)

    if path.exists():
        size_kb = path.stat().st_size / 1024
        elapsed = time.perf_counter() - t0_save
        logger.success(f"  saved {path.name} ({size_kb:.1f} KB) in {elapsed:.2f}s.")
        return True

    logger.error(f"  file not found after save attempt: {path}.")
    return False


def close_window(window_substring: str) -> None:
    def _dismiss_save_dialog() -> None:
        pyautogui.hotkey("alt", "n")
        time.sleep(0.15)
        if "save" in active_window_title().lower():
            pyautogui.press("tab")
            time.sleep(0.1)
            pyautogui.press("enter")
            time.sleep(0.15)

    for win in windows_with_title(window_substring):
        try:
            win.activate()
            time.sleep(0.15)
            pyautogui.hotkey("alt", "f4")
            time.sleep(0.35)
            prompt = active_window_title().lower()
            if "save" in prompt:
                _dismiss_save_dialog()
            time.sleep(0.2)
            if windows_with_title(window_substring):
                pyautogui.hotkey("alt", "f4")
                time.sleep(0.3)
                if "save" in active_window_title().lower():
                    _dismiss_save_dialog()
        except Exception:  # noqa: BLE001
            continue
    logger.info(f"  closed '{window_substring}' windows.")
