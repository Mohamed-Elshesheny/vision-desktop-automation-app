# Vision-Based Desktop Automation with Dynamic Icon Grounding

A Windows desktop automation agent that **locates the Notepad icon without fixed coordinates**, opens it, fetches blog posts from a remote API, and saves each post as a text file. Grounding is **resolution-independent**: the same pipeline works when the icon is moved anywhere on the desktop.

Layered fallbacks and production-minded automation (retries, verification, logging, offline resilience).

> **Caution — VM + demo video:** This project was developed and tested on a **Windows virtual machine**, not on bare-metal hardware. The demo video was recorded with **OBS** inside that VM; OBS added CPU/GPU overhead, so recording quality was lowered to keep the system responsive. **Performance in the video is not representative of normal runs** — on a real Windows machine (without VM overhead or screen recording), grounding and the full workflow should run noticeably faster.

---

## Overview

The workflow processes **10 blog posts**:

1. Fetch posts from JSONPlaceholder (falls back to bundled JSON if offline).
2. Minimize windows and capture a fresh desktop screenshot.
3. Ground the **Notepad** desktop shortcut via a cascaded vision pipeline.
4. Double-click, paste content, save as `post_N.txt`, close Notepad.
5. Retry failed posts with spatial exclusion and a warmed template cache.

**Stack:** Python 3.12 · [uv](https://github.com/astral-sh/uv) · GPT-4o · OpenCV · RapidOCR · mss · pyautogui · pydantic-settings · loguru

---

## Grounding Pipeline

| Layer | Method | Typical latency |
|-------|--------|-----------------|
| 1 | **Template cache** — ORB + Canny match against last successful crop | ~1–2 s |
| 2 | **OCR fast path** — RapidOCR exact label on desktop columns | ~2–8 s |
| 3 | **VLM grid** — GPT-4o picks a 6×6 cell, OCR anchors + crosshair inside cell | ~15–30 s |

Candidates pass **GPT-4o verify** (skipped for template confidence ≥ 0.85). Failed coords and grid cells are excluded on retry.

---

## Visual Grounding Results

Notepad was moved to three desktop regions. Each screenshot was generated with `demo --tag` after moving the shortcut. **Green box** = detected region · **Red crosshair** = click point (`click_xy`, or box center for template-only hits).

| Region | Result |
|--------|--------|
| **Top-left** | ![Top-left grounding](docs/deliverables/grounding_top_left.png) |
| **Center** | ![Center grounding](docs/deliverables/grounding_center.png) |
| **Bottom-right** | ![Bottom-right grounding](docs/deliverables/grounding_bottom_right.png) |

```powershell
uv run vision-desktop-automation demo --tag top_left
uv run vision-desktop-automation demo --tag center
uv run vision-desktop-automation demo --tag bottom_right
```

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Windows 10/11 | Target OS |
| 1920×1080 @ 100% scale | Recommended |
| Notepad shortcut on desktop | Label must read **Notepad** |
| OpenAI API key | GPT-4o for grid, crosshair, verify |
| [uv](https://github.com/astral-sh/uv) | Install and run dependencies |

---

## Quick Start

```powershell
git clone <repo-url>
cd vision-desktop-automation
uv sync
copy .env.example .env   # set OPENAI_API_KEY=sk-...
uv run vision-desktop-automation
```

Output: `Desktop\tjm-project\post_*.txt` · Logs: `output/logs/` · Annotations: `output/annotations/`

---

## Commands

| Command | Description |
|---------|-------------|
| `uv run vision-desktop-automation` | Full 10-post workflow + retry pass |
| `uv run vision-desktop-automation demo --tag NAME` | Ground once → `docs/deliverables/grounding_<tag>.png` |

---

## Configuration

Settings: `src/vision_desktop_automation/config.py`. Required in `.env`:

```env
OPENAI_API_KEY=sk-...
```

Optional: `ICON_LABEL`, `ICON_OFFSET_PX`, `OCR_SCAN_RIGHT_COLUMN`, `MAX_RETRIES`, `SAVE_DIR`, `OUTPUT_DIR`, etc. (uppercase field names; defaults apply if omitted).

`ICON_OFFSET_PX` (default `32`) sets how far above an OCR label the click lands; it scales with screen height. `OCR_SCAN_RIGHT_COLUMN=true` also scans the right desktop column (~26% width) for icons placed there.

---

## Project Structure

```
src/vision_desktop_automation/
├── main.py · workflow.py · config.py · api.py · automation.py
├── grounding/          coarse_to_fine · ocr_locate · template_cache · providers/openai
└── vision/             screenshot · preprocess · annotate
docs/deliverables/      annotated grounding screenshots (above)
```

---

## Design Highlights

- **No hardcoded coordinates** — discovery each run; template cache is an optimization.
- **Self-healing cache** — updated on success, invalidated on verification failure.
- **Offline fallback** — `data/fallback_posts.json` when the API is unreachable.
- **DPI-aware capture** — mss screenshots and pyautogui clicks stay aligned on scaled displays.

---

## Author

Mohamed Elshesheny
