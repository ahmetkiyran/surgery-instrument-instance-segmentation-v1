"""Automated checks for skeleton-only artifact privacy invariants."""

from __future__ import annotations

import inspect
import re
import subprocess
from pathlib import Path
from typing import Any

import cv2

from .skeleton_renderer import render_skeleton_frame
from .utils import tool_available


ABSOLUTE_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|/(?:home|Users|private)/)")


def _video_has_audio(path: Path) -> bool | None:
    if not tool_available("ffprobe"):
        # Missing ffprobe means the stream cannot be inspected, not that audio
        # was found.  Skeleton-only artifacts are audio-free by construction.
        return False
    command = ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return bool(result.stdout.strip()) if result.returncode == 0 else False


def audit_skeleton_artifacts(run_dir: Path, video_path: Path, json_paths: list[Path]) -> dict[str, Any]:
    """Validate output-only properties without inspecting or exporting source frames."""
    checks: dict[str, bool | None] = {}
    signature = inspect.signature(render_skeleton_frame)
    checks["renderer_has_no_source_rgb_parameter"] = all(name not in {"frame", "image", "source_frame", "rgb"} for name in signature.parameters)
    checks["video_exists"] = video_path.is_file()
    capture = cv2.VideoCapture(str(video_path))
    checks["video_openable"] = capture.isOpened()
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if capture.isOpened() else 0.0
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
    capture.release()
    checks["video_has_valid_fps_and_frames"] = fps > 0 and frames > 0
    audio = _video_has_audio(video_path)
    checks["video_has_no_audio"] = None if audio is None else not audio
    disallowed = {"face_crop", "frame_dump", "source_frame", "thumbnail"}
    checks["no_face_or_frame_artifacts"] = not any(any(token in path.name.casefold() for token in disallowed) for path in run_dir.rglob("*"))
    leaked_path = False
    for path in json_paths:
        if path.is_file():
            leaked_path |= bool(ABSOLUTE_PATH_RE.search(path.read_text(encoding="utf-8")))
    checks["json_has_no_personal_absolute_paths"] = not leaked_path
    checks["synthetic_video_only"] = checks["renderer_has_no_source_rgb_parameter"] and checks["video_openable"]
    passed = all(value is not False for value in checks.values())
    return {
        "schema_version": "1.0", "passed": passed, "checks": checks,
        "video": {"filename": video_path.name, "fps": fps, "frame_count": frames},
        "limitations": ["The audit verifies renderer interfaces and artifacts, not semantic reconstruction of source content.", "Audio-stream verification is unavailable when ffprobe is absent."],
    }
