"""Relative 3D coordinate generation and robust trajectory distance metrics."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .schemas import Detection, TrackPoint


def robust_mask_depth(depth_map: np.ndarray, mask: np.ndarray) -> float | None:
    values = depth_map[(mask > 0) & np.isfinite(depth_map)]
    if values.size < 3:
        return None
    low, high = np.percentile(values, [5, 95])
    cleaned = values[(values >= low) & (values <= high)]
    return float(np.median(cleaned)) if cleaned.size else None


def point_from_detection(detection: Detection, depth_map: np.ndarray | None, frame_width: int, frame_height: int) -> TrackPoint | None:
    if detection.track_id is None:
        return None
    centroid = detection.centroid()
    if centroid is None:
        return None
    x, y = centroid
    depth = robust_mask_depth(depth_map, detection.mask) if depth_map is not None else None
    return TrackPoint(
        frame_index=detection.frame_index,
        timestamp_s=detection.timestamp_s,
        track_id=detection.track_id,
        class_name=detection.class_name,
        confidence=detection.confidence,
        x=x,
        y=y,
        depth=depth,
        relative_x=x / max(frame_width - 1, 1),
        relative_y=y / max(frame_height - 1, 1),
        relative_depth=depth,
        depth_valid=depth is not None,
    )


def smooth_points(points: list[TrackPoint], window: int) -> list[TrackPoint]:
    """Apply a causal rolling median separately to each track's relative coordinates."""
    by_track: dict[tuple[str, int], list[TrackPoint]] = defaultdict(list)
    for point in points:
        by_track[(point.class_name, point.track_id)].append(point)
    for track_points in by_track.values():
        track_points.sort(key=lambda item: item.timestamp_s)
        for index, point in enumerate(track_points):
            current = track_points[max(0, index - window + 1) : index + 1]
            for attribute in ("relative_x", "relative_y", "relative_depth"):
                values = [getattr(item, attribute) for item in current if getattr(item, attribute) is not None]
                if values:
                    setattr(point, attribute, float(np.median(values)))
    return points


def trajectory_lengths(points: list[TrackPoint], max_jump: float) -> dict[tuple[str, int], dict[str, float | int]]:
    """Return 2D/relative-3D path lengths while excluding implausibly large discontinuities."""
    by_track: dict[tuple[str, int], list[TrackPoint]] = defaultdict(list)
    for point in points:
        by_track[(point.class_name, point.track_id)].append(point)
    result: dict[tuple[str, int], dict[str, float | int]] = {}
    for key, track_points in by_track.items():
        track_points.sort(key=lambda item: item.timestamp_s)
        distance_2d = distance_3d = 0.0
        valid_segments = 0
        for previous, current in zip(track_points, track_points[1:], strict=False):
            if None in (previous.relative_x, previous.relative_y, current.relative_x, current.relative_y):
                continue
            delta_2d = float(np.hypot(current.relative_x - previous.relative_x, current.relative_y - previous.relative_y))
            if delta_2d <= max_jump:
                distance_2d += delta_2d
                if previous.relative_depth is not None and current.relative_depth is not None:
                    delta_3d = float(np.linalg.norm([current.relative_x - previous.relative_x, current.relative_y - previous.relative_y, current.relative_depth - previous.relative_depth]))
                    if delta_3d <= max_jump:
                        distance_3d += delta_3d
                valid_segments += 1
        valid_depth = sum(point.depth_valid for point in track_points)
        result[key] = {"relative_2d_motion": distance_2d, "relative_3d_motion": distance_3d, "sample_count": len(track_points), "depth_validity_ratio": valid_depth / len(track_points) if track_points else 0.0, "valid_segments": valid_segments}
    return result
