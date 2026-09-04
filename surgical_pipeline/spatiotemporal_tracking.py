"""Relative 4D records for confident pose keypoints; never fabricates depth."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .pose_schemas import PoseObservation, PosePoint4D


def robust_point_depth(depth_map: np.ndarray | None, x: float, y: float, radius_px: int = 3) -> float | None:
    if depth_map is None or depth_map.ndim != 2:
        return None
    height, width = depth_map.shape
    left, right = max(0, int(round(x)) - radius_px), min(width, int(round(x)) + radius_px + 1)
    top, bottom = max(0, int(round(y)) - radius_px), min(height, int(round(y)) + radius_px + 1)
    values = depth_map[top:bottom, left:right]
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return None
    low, high = np.percentile(finite, [5, 95])
    cleaned = finite[(finite >= low) & (finite <= high)]
    return float(np.median(cleaned)) if cleaned.size else None


def pose_points_4d(pose: PoseObservation, frame_width: int, frame_height: int, depth_map: np.ndarray | None, keypoint_names: tuple[str, ...], confidence_threshold: float = 0.25) -> list[PosePoint4D]:
    if pose.pose_track_id is None:
        return []
    result: list[PosePoint4D] = []
    for index in pose.visible_indices(confidence_threshold):
        x, y, confidence = (float(value) for value in pose.keypoints[index, :3])
        z_rel = robust_point_depth(depth_map, x, y)
        result.append(PosePoint4D(
            pose_track_id=pose.pose_track_id, keypoint_index=int(index), keypoint_name=keypoint_names[int(index)] if int(index) < len(keypoint_names) else f"keypoint_{index}",
            frame_index=pose.frame_index, timestamp_s=pose.timestamp_s, x=x, y=y, confidence=confidence,
            x_rel=(x - (frame_width - 1) / 2.0) / max((frame_width - 1) / 2.0, 1.0),
            y_rel=(y - (frame_height - 1) / 2.0) / max((frame_height - 1) / 2.0, 1.0),
            z_rel=z_rel, depth_valid=z_rel is not None, depth_source="robust_neighbourhood_median" if z_rel is not None else "unavailable",
        ))
    return result


def smooth_pose_points(points: list[PosePoint4D], window: int = 3) -> list[PosePoint4D]:
    """Apply a causal rolling median to available relative 4D geometry only."""
    if window < 1:
        raise ValueError("Pose smoothing window en az 1 olmalıdır.")
    grouped: dict[tuple[int, int], list[PosePoint4D]] = defaultdict(list)
    for point in points:
        grouped[(point.pose_track_id, point.keypoint_index)].append(point)
    for series in grouped.values():
        series.sort(key=lambda item: item.timestamp_s)
        for index, point in enumerate(series):
            history = series[max(0, index - window + 1) : index + 1]
            point.x_rel = float(np.median([item.x_rel for item in history]))
            point.y_rel = float(np.median([item.y_rel for item in history]))
            depth = [item.z_rel for item in history if item.z_rel is not None]
            if depth:
                point.z_rel = float(np.median(depth))
    return points


def instrument_point_4d(detection, frame_width: int, frame_height: int, depth_map: np.ndarray | None) -> dict | None:
    centre = detection.centroid()
    if centre is None or detection.track_id is None:
        return None
    x, y = centre
    z_rel = robust_point_depth(depth_map, x, y)
    return {
        "schema_version": "1.0", "track_id": detection.track_id, "class_id": detection.class_id,
        "class_name": detection.class_name, "frame_index": detection.frame_index, "timestamp_s": detection.timestamp_s,
        "confidence": detection.confidence, "bbox_xyxy": list(detection.bbox_xyxy) if detection.bbox_xyxy is not None else None,
        "mask_area_px": int((detection.mask > 0).sum()),
        "center_x": x, "center_y": y, "x_rel": (x - (frame_width - 1) / 2.0) / max((frame_width - 1) / 2.0, 1.0),
        "y_rel": (y - (frame_height - 1) / 2.0) / max((frame_height - 1) / 2.0, 1.0),
        "z_rel": z_rel, "depth_valid": z_rel is not None, "centre_source": detection.centroid_source(),
    }
