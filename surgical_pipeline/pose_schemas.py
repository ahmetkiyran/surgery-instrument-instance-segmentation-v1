"""Typed, privacy-safe data contracts for the optional skeleton-only pose pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


COCO17_KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
COCO17_SKELETON = (
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6),
)


@dataclass(slots=True)
class PoseObservation:
    bbox_xyxy: tuple[float, float, float, float]
    keypoints: np.ndarray
    confidence: float
    frame_index: int
    timestamp_s: float
    health_match_status: str = "unmatched_person"
    health_match_score: float = 0.0
    pose_track_id: int | None = None
    tracking_status: str = "unassigned"

    def visible_indices(self, threshold: float) -> np.ndarray:
        if self.keypoints.ndim != 2 or self.keypoints.shape[1] < 3:
            return np.asarray([], dtype=int)
        return np.flatnonzero(np.isfinite(self.keypoints[:, :3]).all(axis=1) & (self.keypoints[:, 2] >= threshold))

    def centre(self, threshold: float = 0.0) -> tuple[float, float]:
        visible = self.visible_indices(threshold)
        if visible.size:
            points = self.keypoints[visible, :2]
            return float(np.median(points[:, 0])), float(np.median(points[:, 1]))
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def row_2d(self) -> dict[str, Any]:
        centre_x, centre_y = self.centre(0.0)
        return {
            "schema_version": "1.0", "pose_track_id": self.pose_track_id, "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s, "bbox_x1": self.bbox_xyxy[0], "bbox_y1": self.bbox_xyxy[1],
            "bbox_x2": self.bbox_xyxy[2], "bbox_y2": self.bbox_xyxy[3], "pose_confidence": self.confidence,
            "visible_keypoint_count": int(self.visible_indices(0.0).size), "health_match_status": self.health_match_status,
            "health_match_score": self.health_match_score, "tracking_status": self.tracking_status,
            "center_x": centre_x, "center_y": centre_y,
        }


@dataclass(slots=True)
class PosePoint4D:
    pose_track_id: int
    keypoint_index: int
    keypoint_name: str
    frame_index: int
    timestamp_s: float
    x: float
    y: float
    confidence: float
    x_rel: float
    y_rel: float
    z_rel: float | None
    depth_valid: bool
    depth_source: str

    def row(self) -> dict[str, Any]:
        return {"schema_version": "1.0", **asdict(self)}


@dataclass(frozen=True)
class PoseModelInfo:
    path: str
    sha256: str
    task: str
    names: dict[int, str]
    keypoint_shape: tuple[int, int]
    keypoint_names: tuple[str, ...]

    def safe_dict(self) -> dict[str, Any]:
        return {
            "filename": self.path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1], "sha256": self.sha256,
            "task": self.task, "classes": self.names, "keypoint_shape": self.keypoint_shape,
            "keypoint_names": list(self.keypoint_names), "skeleton_connections": [list(edge) for edge in COCO17_SKELETON],
        }


@dataclass
class PoseQuality:
    processed_frames: int = 0
    frames_with_pose: int = 0
    total_poses: int = 0
    total_visible_keypoints: int = 0
    total_keypoint_confidence: float = 0.0
    matched_poses: int = 0
    unmatched_poses: int = 0
    full_skeleton_poses: int = 0
    partial_skeleton_poses: int = 0
    joint_visible_counts: dict[str, int] = field(default_factory=dict)
    track_loss_count: int = 0
    id_switch_estimate: int = 0
    jitter_samples: list[float] = field(default_factory=list)

    def add(self, poses: list[PoseObservation], threshold: float, previous_centres: dict[int, tuple[float, float]]) -> None:
        self.processed_frames += 1
        self.frames_with_pose += int(bool(poses))
        self.total_poses += len(poses)
        for pose in poses:
            visible = pose.visible_indices(threshold)
            self.total_visible_keypoints += int(visible.size)
            if visible.size >= 10:
                self.full_skeleton_poses += 1
            elif visible.size:
                self.partial_skeleton_poses += 1
            if visible.size:
                self.total_keypoint_confidence += float(np.sum(pose.keypoints[visible, 2]))
            for index in visible:
                name = COCO17_KEYPOINT_NAMES[int(index)] if int(index) < len(COCO17_KEYPOINT_NAMES) else f"keypoint_{index}"
                self.joint_visible_counts[name] = self.joint_visible_counts.get(name, 0) + 1
            if pose.health_match_status == "health_personnel":
                self.matched_poses += 1
            else:
                self.unmatched_poses += 1
            if pose.pose_track_id is not None:
                centre = pose.centre(threshold)
                previous = previous_centres.get(pose.pose_track_id)
                if previous is not None:
                    self.jitter_samples.append(float(np.hypot(centre[0] - previous[0], centre[1] - previous[1])))
                previous_centres[pose.pose_track_id] = centre

    def report(self, track_lengths: dict[int, int]) -> dict[str, Any]:
        pose_count = max(self.total_poses, 1)
        visibility = {name: self.joint_visible_counts.get(name, 0) / pose_count for name in COCO17_KEYPOINT_NAMES}
        pair_ratio = lambda names: sum(visibility[name] for name in names) / len(names)
        return {
            "schema_version": "1.0", "processed_frames": self.processed_frames,
            "pose_found_frame_ratio": self.frames_with_pose / max(self.processed_frames, 1),
            "mean_poses_per_frame": self.total_poses / max(self.processed_frames, 1),
            "mean_visible_keypoints_per_pose": self.total_visible_keypoints / pose_count,
            "mean_keypoint_confidence": self.total_keypoint_confidence / max(self.total_visible_keypoints, 1),
            "joint_visibility_ratio": visibility,
            "shoulder_visibility_ratio": pair_ratio(("left_shoulder", "right_shoulder")),
            "elbow_visibility_ratio": pair_ratio(("left_elbow", "right_elbow")),
            "wrist_visibility_ratio": pair_ratio(("left_wrist", "right_wrist")),
            "hip_visibility_ratio": pair_ratio(("left_hip", "right_hip")),
            "full_skeleton_ratio": self.full_skeleton_poses / pose_count,
            "partial_skeleton_ratio": self.partial_skeleton_poses / pose_count,
            "health_matched_pose_ratio": self.matched_poses / pose_count, "unmatched_pose_count": self.unmatched_poses,
            "mean_track_length_frames": sum(track_lengths.values()) / max(len(track_lengths), 1),
            "short_track_loss_count": self.track_loss_count, "estimated_id_switch_count": self.id_switch_estimate,
            "mean_center_jitter_px": float(np.mean(self.jitter_samples)) if self.jitter_samples else None,
            "note": "These are temporal/structural quality indicators only; no keypoint ground truth, precision, recall, or mAP is claimed.",
        }
