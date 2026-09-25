"""Artifact discovery restricted to a single server-created job directory."""

from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass
from pathlib import Path

from .schemas import ArtifactResponse
from .security import ensure_child


ALLOWED_SUFFIXES = {".mp4", ".json", ".csv", ".html", ".png", ".zip", ".yaml", ".yml"}


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    path: Path
    filename: str
    kind: str
    media_type: str
    sensitive: bool

    def response(self) -> ArtifactResponse:
        return ArtifactResponse(
            artifact_id=self.artifact_id,
            filename=self.filename,
            kind=self.kind,
            size_bytes=self.path.stat().st_size,
            media_type=self.media_type,
            sensitive=self.sensitive,
        )


def _kind(path: Path, render_mode: str | None = None) -> str:
    name = path.name.casefold()
    if name == "sam3_inspection.mp4":
        return "inspection_video"
    if name in {"inspection_tracking.mp4", "tracking_rgb.mp4"}:
        return "inspection_video"
    if name == "processed_video_xray_sam3.mp4":
        return "privacy_xray_video"
    if name in {"privacy_xray_tracking.mp4", "security_xray_final.mp4"}:
        return "privacy_xray_video"
    if name == "skeleton_tracking.mp4":
        return "skeleton_video"
    if name == "processed_video.mp4" and render_mode in {"dual", "privacy-xray"}:
        return "privacy_xray_video"
    if name == "processed_video.mp4":
        return "blur_video"
    if "trajectory" in name and path.suffix == ".html":
        return "relative_4d_html"
    if "usage" in name:
        return "instrument_usage"
    if "pose" in name:
        return "personnel_motion"
    if path.suffix == ".json":
        return "summary"
    return "file"


def _media_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def collect_artifacts(job_root: Path) -> dict[str, Artifact]:
    root = job_root.resolve()
    result: dict[str, Artifact] = {}
    if not root.is_dir():
        return result
    render_mode: str | None = None
    config_path = root / "run_config.yaml"
    if config_path.is_file():
        for line in config_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("render_mode:"):
                render_mode = line.partition(":")[2].strip()
                break
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file() or candidate.suffix.casefold() not in ALLOWED_SUFFIXES:
            continue
        path = ensure_child(candidate, root)
        # The core's blur run can create a convenience archive containing the
        # blurred source-derived video. Do not offer it as a public package.
        if path.suffix.casefold() == ".zip" and (path.parent / "processed_video.mp4").is_file():
            continue
        relative = path.relative_to(root).as_posix()
        artifact_id = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:24]
        kind = _kind(path, render_mode)
        result[artifact_id] = Artifact(
            artifact_id=artifact_id,
            path=path,
            filename=path.name,
            kind=kind,
            media_type=_media_type(path),
            # Inspection/RGB outputs contain the source scene and therefore
            # require the same explicit confirmation as other source-derived
            # media.  Synthetic X-ray output is geometry-only.
            sensitive=kind in {"blur_video", "inspection_video"},
        )
    return result
