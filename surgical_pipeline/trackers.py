"""Per-model Ultralytics tracker configuration with no shared tracker state."""

from __future__ import annotations

from pathlib import Path

import yaml

from .config import AppConfig


def write_tracker_config(path: Path, config: AppConfig, confidence: float | None = None, iou: float | None = None) -> Path:
    """Write a standard Ultralytics BoT-SORT/ByteTrack config for one model instance."""
    confidence = config.confidence if confidence is None else confidence
    iou = config.iou if iou is None else iou
    if config.tracker_algorithm == "botsort":
        values = {
            "tracker_type": "botsort",
            "track_high_thresh": confidence,
            "track_low_thresh": max(0.01, confidence * 0.45),
            "new_track_thresh": confidence,
            "track_buffer": config.track_buffer,
            "match_thresh": iou,
            "fuse_score": True,
            "gmc_method": "sparseOptFlow",
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.25,
            "with_reid": False,
            # Required by current Ultralytics even when ReID is disabled.
            "model": "auto",
        }
    else:
        values = {
            "tracker_type": "bytetrack",
            "track_high_thresh": confidence,
            "track_low_thresh": max(0.01, confidence * 0.45),
            "new_track_thresh": confidence,
            "track_buffer": config.track_buffer,
            "match_thresh": iou,
            "fuse_score": True,
        }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(values, handle, sort_keys=False)
    return path


def tracked_result(model, frame, tracker_path: Path, confidence: float, iou: float, device: str, use_half: bool):
    """Call track with persist=True; each YOLO object owns a separate tracker lifecycle."""
    result = model.track(
        source=frame,
        persist=True,
        tracker=str(tracker_path),
        conf=confidence,
        iou=iou,
        device=device,
        half=use_half,
        verbose=False,
    )
    return result[0]
