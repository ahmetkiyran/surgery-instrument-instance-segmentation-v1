"""Purely synthetic renderer: it accepts geometry, never source RGB pixels."""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from .pose_schemas import COCO17_SKELETON, PoseObservation
from .schemas import Detection


INSTRUMENT_COLOURS = ((80, 190, 255), (255, 145, 80), (190, 100, 255), (90, 220, 150))
HEALTH_COLOUR = (90, 230, 120)
UNMATCHED_COLOUR = (175, 175, 175)


def _instrument_centre(detection: Detection) -> tuple[int, int] | None:
    centre = detection.centroid()
    return (int(round(centre[0])), int(round(centre[1]))) if centre is not None else None


def render_skeleton_frame(
    frame_size: tuple[int, int],
    timestamp: float,
    poses: list[PoseObservation],
    instruments: list[Detection],
    personnel_count: int,
    trajectories: dict[tuple[str, int], list[tuple[int, int]]] | None = None,
    keypoint_confidence: float = 0.25,
    show_pose_ids: bool = False,
    grid: bool = True,
) -> np.ndarray:
    """Create a new synthetic frame without accepting or reading a source image."""
    width, height = frame_size
    canvas = np.full((height, width, 3), 16, dtype=np.uint8)
    if grid:
        for x in range(0, width, max(32, width // 16)):
            cv2.line(canvas, (x, 0), (x, height), (35, 35, 35), 1)
        for y in range(0, height, max(32, height // 12)):
            cv2.line(canvas, (0, y), (width, y), (35, 35, 35), 1)
    for pose in poses:
        colour = HEALTH_COLOUR if pose.health_match_status == "health_personnel" else UNMATCHED_COLOUR
        visible = set(int(index) for index in pose.visible_indices(keypoint_confidence))
        for left, right in COCO17_SKELETON:
            if left in visible and right in visible:
                start = tuple(np.rint(pose.keypoints[left, :2]).astype(int))
                end = tuple(np.rint(pose.keypoints[right, :2]).astype(int))
                cv2.line(canvas, start, end, colour, 2, cv2.LINE_AA)
        for index in visible:
            point = tuple(np.rint(pose.keypoints[index, :2]).astype(int))
            cv2.circle(canvas, point, 3, colour, -1, cv2.LINE_AA)
        if show_pose_ids and pose.pose_track_id is not None:
            centre = tuple(int(round(value)) for value in pose.centre(keypoint_confidence))
            cv2.putText(canvas, f"P{pose.pose_track_id:02d}", centre, cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    trail_map = trajectories if trajectories is not None else defaultdict(list)
    for detection in instruments:
        centre = _instrument_centre(detection)
        if centre is None:
            continue
        key = (detection.class_name, detection.track_id or -1)
        points = trail_map.get(key, [])
        colour = INSTRUMENT_COLOURS[sum(detection.class_name.encode("utf-8")) % len(INSTRUMENT_COLOURS)]
        if len(points) > 1:
            cv2.polylines(canvas, [np.asarray(points[-20:], dtype=np.int32)], False, colour, 1, cv2.LINE_AA)
        cv2.circle(canvas, centre, 5, colour, -1, cv2.LINE_AA)
        cv2.putText(canvas, detection.class_name, (min(max(0, centre[0] + 6), max(0, width - 5)), max(14, centre[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Health personnel: {personnel_count}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"t={timestamp:.2f}s | relative motion display", (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1, cv2.LINE_AA)
    return canvas
