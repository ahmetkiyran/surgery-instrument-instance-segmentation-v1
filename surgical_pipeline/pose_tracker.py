"""Health-personnel pose matching and lightweight anonymous pose tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pose_schemas import PoseObservation
from .schemas import Detection


def _bbox_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if not area:
        return 0.0
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return area / max(left_area + right_area - area, 1e-9)


def match_health_personnel(poses: list[PoseObservation], health_detections: list[Detection], keypoint_confidence: float = 0.25) -> list[PoseObservation]:
    """Tag only an evidenced health match; unmatched poses are never called patients."""
    for pose in poses:
        best = 0.0
        centre = pose.centre(keypoint_confidence)
        visible = pose.visible_indices(keypoint_confidence)
        for health in health_detections:
            bbox = health.bbox_xyxy
            if bbox is None:
                ys, xs = np.where(health.mask > 0)
                if len(xs):
                    bbox = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
            if bbox is None:
                continue
            iou = _bbox_iou(pose.bbox_xyxy, bbox)
            diagonal = max(float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])), 1.0)
            centre_score = max(0.0, 1.0 - float(np.hypot(centre[0] - (bbox[0] + bbox[2]) / 2, centre[1] - (bbox[1] + bbox[3]) / 2)) / diagonal)
            inside_ratio = 0.0
            if visible.size and health.mask.size:
                xs = np.clip(np.rint(pose.keypoints[visible, 0]).astype(int), 0, health.mask.shape[1] - 1)
                ys = np.clip(np.rint(pose.keypoints[visible, 1]).astype(int), 0, health.mask.shape[0] - 1)
                inside_ratio = float(np.mean(health.mask[ys, xs] > 0))
            best = max(best, 0.55 * iou + 0.30 * inside_ratio + 0.15 * centre_score)
        pose.health_match_score = best
        pose.health_match_status = "health_personnel" if best >= 0.15 else "unmatched_person"
    return poses


@dataclass
class _Track:
    bbox: tuple[float, float, float, float]
    centre: tuple[float, float]
    last_frame: int
    observations: int = 0
    missed: int = 0


class PoseTracker:
    """IoU/centre/keypoint-compatible temporal association without identity semantics."""

    def __init__(self, max_lost_frames: int = 8, match_threshold: float = 0.25) -> None:
        self.max_lost_frames = max_lost_frames
        self.match_threshold = match_threshold
        self._next_id = 1
        self._tracks: dict[int, _Track] = {}
        self.track_lengths: dict[int, int] = {}
        self.short_loss_count = 0
        self.id_switch_estimate = 0

    def update(self, poses: list[PoseObservation], frame_index: int, keypoint_confidence: float = 0.25) -> list[PoseObservation]:
        assigned: set[int] = set()
        for pose in poses:
            centre = pose.centre(keypoint_confidence)
            candidates: list[tuple[float, int]] = []
            for track_id, track in self._tracks.items():
                if track_id in assigned:
                    continue
                distance = float(np.hypot(centre[0] - track.centre[0], centre[1] - track.centre[1]))
                scale = max(np.hypot(track.bbox[2] - track.bbox[0], track.bbox[3] - track.bbox[1]), 1.0)
                score = 0.7 * _bbox_iou(pose.bbox_xyxy, track.bbox) + 0.3 * max(0.0, 1.0 - distance / scale)
                candidates.append((score, track_id))
            score, track_id = max(candidates, default=(0.0, -1))
            if score < self.match_threshold:
                track_id = self._next_id
                self._next_id += 1
                status = "new"
            else:
                status = "continued" if self._tracks[track_id].missed == 0 else "reacquired"
                if self._tracks[track_id].missed:
                    self.short_loss_count += 1
            previous = self._tracks.get(track_id)
            self._tracks[track_id] = _Track(pose.bbox_xyxy, centre, frame_index, (previous.observations + 1) if previous else 1, 0)
            self.track_lengths[track_id] = self.track_lengths.get(track_id, 0) + 1
            pose.pose_track_id = track_id
            pose.tracking_status = status
            assigned.add(track_id)
        for track_id, track in list(self._tracks.items()):
            if track_id in assigned:
                continue
            track.missed += 1
            if track.missed > self.max_lost_frames:
                del self._tracks[track_id]
        return poses
