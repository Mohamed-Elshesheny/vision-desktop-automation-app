from __future__ import annotations

GROUND_SYSTEM = (
    "You are a precise GUI visual grounding engine. You look at a screenshot and "
    "return the pixel regions of UI targets. You never invent elements that are "
    "not visible. You always answer with strict JSON and nothing else."
)


def coarse_grid_prompt(icon_label: str, excluded_cells: list[str] | None = None) -> str:
    exclusion_note = ""
    if excluded_cells:
        cells_str = ", ".join(excluded_cells)
        exclusion_note = f"""
⚠ IMPORTANT — PREVIOUSLY TRIED AND FAILED:
Cell(s) {cells_str} have already been clicked but did NOT open the target
application. There must be a DIFFERENT icon elsewhere. Do NOT return any of
these excluded cells. Search the rest of the grid carefully.
"""
    return f"""You are a visual grounding assistant. The image is a Windows desktop screenshot with a labeled grid overlay.

The grid divides the screen into cells labeled A1 through F6:
  - Rows A to F from top to bottom
  - Columns 1 to 6 from left to right
{exclusion_note}
Task: Find the **{icon_label}** desktop icon and identify which cell it is in.

IMPORTANT rules:
1. Scan the ENTIRE grid — the icon can be anywhere, including CENTER cells (C3, C4, D3, D4) or corners. Do NOT assume it is at the edges.
2. A desktop icon is a small square graphic (32–64 px) with a short text label DIRECTLY BELOW it. It is NOT part of the wallpaper photo.
3. The wallpaper is a background photo — do NOT identify any part of the photo as an icon.
4. Report the cell containing the CENTER of the icon graphic.
5. If the icon is near a cell boundary, also report the best alternative in "backup_cell".
6. If you cannot find the icon anywhere, set "cell" to "NONE" and "backup_cell" to "NONE".
7. Rate your confidence from 0.0 to 1.0 in "confidence".

Respond ONLY with JSON (no extra text):
{{"cell": "<label or NONE>", "backup_cell": "<label or NONE>", "confidence": <0.0-1.0>}}"""


def point_grid_prompt(icon_label: str) -> str:
    return f"""You are a visual grounding assistant. You are given a CROPPED region of a Windows desktop.
Numbered magenta crosshairs have been drawn over the image, marking the center of potential icons.

Your task: Find the **{icon_label}** desktop icon graphic, and report the point ID of the crosshair that is closest to its exact center.

Rules:
1. Find the icon graphic for "{icon_label}".
2. Identify the single magenta crosshair that is positioned nearest to the CENTER of that icon.
3. If multiple matching icons exist, pick the one that represents the main application (avoid shortcuts with arrows if possible, and avoid applications with similar but distinct names like 'Notepad++' unless explicitly asked for).
4. Report the numeric ID of that point (0-based).
5. If you cannot find the icon, respond with {{"point_id": -1}}.

Respond with ONLY a JSON object (no extra text):
{{"point_id": <int>}}"""


def verify_prompt(icon_label: str) -> str:
    return f"""This is a small crop centered on where a detector believes the Windows
desktop shortcut icon "{icon_label}" is located.

Answer ONLY "match: true" if ALL of the following conditions are met:
  1. A small SQUARE icon graphic (application logo / thumbnail) is visible
     near the CENTER of this crop.
  2. A short text label that reads EXACTLY "{icon_label}" — no extra words or
     suffixes (reject "++", "- Copy", "(2)", etc.) — is visible DIRECTLY BELOW
     that graphic.
  3. The icon looks like a genuine Windows desktop shortcut — NOT wallpaper
     art, a photo, a background texture, or any other UI element.

If the crop shows ONLY wallpaper, a window title bar, taskbar icons, text
on a webpage, or any other non-shortcut element, respond with match=false.

Being strict is important: a false positive here causes a mouse click to land
on empty desktop space, which wastes a retry attempt.

Return JSON only: {{"match": true/false, "reasoning": "<one sentence>"}}"""
