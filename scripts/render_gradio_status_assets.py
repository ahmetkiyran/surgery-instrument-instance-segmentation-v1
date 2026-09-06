"""Render tiny animated WebP fallbacks from the repository-owned Lottie status definitions."""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
LOTTIE_ROOT = ROOT / "apps" / "desktop" / "assets" / "lottie"
OUTPUT_ROOT = ROOT / "apps" / "gradio" / "assets" / "status"
STATES = ("idle", "checking_models", "uploading", "analyzing", "finalizing", "completed", "error")


def _find_stroke_colour(value: object) -> tuple[int, int, int]:
    if isinstance(value, dict):
        if value.get("ty") == "st":
            channels = value.get("c", {}).get("k", [])
            if isinstance(channels, list) and len(channels) >= 3:
                return tuple(round(float(channel) * 255) for channel in channels[:3])
        for child in value.values():
            found = _find_stroke_colour(child)
            if found != (45, 202, 213):
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_stroke_colour(child)
            if found != (45, 202, 213):
                return found
    return (45, 202, 213)


def render(state: str) -> None:
    definition = json.loads((LOTTIE_ROOT / f"{state}.json").read_text(encoding="utf-8"))
    colour = _find_stroke_colour(definition)
    frames: list[Image.Image] = []
    for index in range(24):
        image = Image.new("RGBA", (72, 72), (5, 11, 23, 0))
        draw = ImageDraw.Draw(image)
        phase = index / 24
        width = 5
        if state in {"completed", "error", "idle"}:
            radius = 23 + round(2 * math.sin(phase * math.tau))
            draw.ellipse((36 - radius, 36 - radius, 36 + radius, 36 + radius), outline=colour + (255,), width=width)
        else:
            start = round(phase * 360)
            draw.arc((10, 10, 62, 62), start=start, end=start + 245, fill=colour + (255,), width=width)
        if state == "completed":
            draw.line((23, 37, 32, 46, 51, 26), fill=colour + (255,), width=6, joint="curve")
        elif state == "error":
            draw.line((25, 25, 47, 47), fill=colour + (255,), width=6)
            draw.line((47, 25, 25, 47), fill=colour + (255,), width=6)
        elif state == "idle":
            draw.ellipse((31, 31, 41, 41), fill=colour + (255,))
        frames.append(image)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUTPUT_ROOT / f"{state}.webp",
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
        quality=78,
        method=4,
    )


def main() -> int:
    for state in STATES:
        render(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
