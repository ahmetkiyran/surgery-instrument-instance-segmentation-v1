import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from surgical_pipeline.pose_estimator import PoseEstimator, PoseModelError
from surgical_pipeline.pose_schemas import PoseObservation
from surgical_pipeline.pose_tracker import PoseTracker, match_health_personnel
from surgical_pipeline.privacy_audit import audit_skeleton_artifacts
from surgical_pipeline.skeleton_renderer import render_skeleton_frame
from surgical_pipeline.spatiotemporal_tracking import instrument_point_4d, pose_points_4d, smooth_pose_points
from surgical_pipeline.schemas import Detection
from surgical_pipeline.utils import write_json
from surgical_pipeline.video_io import VideoMetadata, VideoWriter


def pose(frame: int = 0, timestamp: float = 0.0, offset: float = 0.0, confidence: float = 0.9) -> PoseObservation:
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[:, 0] = np.linspace(20 + offset, 40 + offset, 17)
    keypoints[:, 1] = np.linspace(10, 35, 17)
    keypoints[:, 2] = confidence
    return PoseObservation((15 + offset, 5, 45 + offset, 40), keypoints, 0.95, frame, timestamp)


def health_detection() -> Detection:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[5:42, 15:46] = 1
    return Detection(0, "health_personel", 0.9, 1, mask, 0, 0.0, (15, 5, 46, 42))


def instrument_detection() -> Detection:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[42:48, 30:39] = 1
    return Detection(0, "scissors", 0.8, 4, mask, 0, 0.0, (30, 42, 39, 48))


def test_pose_model_task_and_keypoint_shape_are_validated_without_real_weight(monkeypatch, tmp_path: Path) -> None:
    weight = tmp_path / "pose.pt"
    weight.write_bytes(b"pose")

    class FakeYOLO:
        def __init__(self, path: str) -> None:
            self.task = "pose"
            self.names = {0: "person"}
            self.ckpt_path = path
            self.model = SimpleNamespace(kpt_shape=[17, 3])

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    info = PoseEstimator(weight, device="cpu").load()
    assert info.task == "pose"
    assert info.keypoint_shape == (17, 3)
    assert len(info.keypoint_names) == 17


def test_non_pose_model_is_rejected(monkeypatch, tmp_path: Path) -> None:
    weight = tmp_path / "not_pose.pt"
    weight.write_bytes(b"not pose")

    class FakeYOLO:
        task = "segment"
        names = {0: "person"}
        model = SimpleNamespace(kpt_shape=[17, 3])
        ckpt_path = str(weight)

        def __init__(self, path: str) -> None:
            pass

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    with pytest.raises(PoseModelError, match="task=segment"):
        PoseEstimator(weight, device="cpu").load()


def test_low_confidence_joint_is_not_rendered_or_emitted_as_4d() -> None:
    item = pose()
    item.keypoints[5, 2] = 0.1
    tracker = PoseTracker()
    tracker.update([item], 0)
    image = render_skeleton_frame((64, 64), 0.0, [item], [], 0, keypoint_confidence=0.25, grid=False)
    blank = render_skeleton_frame((64, 64), 0.0, [], [], 0, keypoint_confidence=0.25, grid=False)
    assert not np.array_equal(image, blank)
    points = pose_points_4d(item, 64, 64, np.ones((64, 64), dtype=np.float32), tuple(f"k{i}" for i in range(17)), 0.25)
    assert all(point.keypoint_index != 5 for point in points)


def test_pose_track_continues_after_short_loss() -> None:
    tracker = PoseTracker(max_lost_frames=2)
    first = tracker.update([pose(0, 0.0)], 0)[0]
    tracker.update([], 1)
    reacquired = tracker.update([pose(2, 0.2, 1.0)], 2)[0]
    assert reacquired.pose_track_id == first.pose_track_id
    assert reacquired.tracking_status == "reacquired"
    assert tracker.short_loss_count == 1


def test_health_match_and_unmatched_person_status() -> None:
    matched, unmatched = pose(), pose(offset=100)
    match_health_personnel([matched, unmatched], [health_detection()])
    assert matched.health_match_status == "health_personnel"
    assert unmatched.health_match_status == "unmatched_person"


def test_instrument_mask_centroid_and_bbox_fallback_are_retained_in_4d() -> None:
    depth = np.full((64, 64), 0.6, dtype=np.float32)
    primary = instrument_point_4d(instrument_detection(), 64, 64, depth)
    fallback = Detection(0, "scissors", 0.8, 5, np.zeros((64, 64), dtype=np.uint8), 0, 0.0, (20, 20, 30, 30))
    secondary = instrument_point_4d(fallback, 64, 64, depth)
    assert primary["centre_source"] == "mask_centroid"
    assert secondary["centre_source"] == "bbox_center"


def test_renderer_cannot_depend_on_source_rgb_pixels() -> None:
    source_a = np.zeros((64, 64, 3), dtype=np.uint8)
    source_b = np.random.default_rng(7).integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    assert not np.array_equal(source_a, source_b)
    tracked = PoseTracker().update([pose()], 0)
    first = render_skeleton_frame((64, 64), 0.0, tracked, [instrument_detection()], 1)
    second = render_skeleton_frame((64, 64), 0.0, tracked, [instrument_detection()], 1)
    assert np.array_equal(first, second)


def test_missing_depth_is_not_zero_and_4d_schema_has_time() -> None:
    item = PoseTracker().update([pose()], 0)[0]
    points = pose_points_4d(item, 64, 64, None, tuple(f"k{i}" for i in range(17)))
    assert points and points[0].z_rel is None and points[0].depth_valid is False
    assert {"x_rel", "y_rel", "z_rel", "timestamp_s"} <= set(points[0].row())


def test_relative_4d_points_are_smoothed_without_fabricating_depth() -> None:
    tracker = PoseTracker()
    first = tracker.update([pose(0, 0.0)], 0)[0]
    second = tracker.update([pose(1, 0.1, 10)], 1)[0]
    depth = np.ones((64, 64), dtype=np.float32)
    points = pose_points_4d(first, 64, 64, depth, tuple(f"k{i}" for i in range(17))) + pose_points_4d(second, 64, 64, depth, tuple(f"k{i}" for i in range(17)))
    smoothed = smooth_pose_points(points, window=2)
    assert smoothed[-1].z_rel is not None


def test_mock_skeleton_video_smoke_and_privacy_audit(tmp_path: Path) -> None:
    metadata = VideoMetadata(64, 64, 10.0, 3, 0.3, False)
    temporary = tmp_path / "silent.mp4"
    writer = VideoWriter(temporary, metadata)
    tracker = PoseTracker()
    for index in range(3):
        tracked = tracker.update(match_health_personnel([pose(index, index / 10, index)], [health_detection()]), index)
        writer.write(render_skeleton_frame((64, 64), index / 10, tracked, [instrument_detection()], 1))
    writer.close()
    video = tmp_path / "skeleton_tracking.mp4"
    writer.finalise(tmp_path / "unused_source.mp4", video, keep_audio=False)
    quality = tmp_path / "pose_quality_report.json"
    manifest = tmp_path / "run_manifest.json"
    write_json(quality, {"schema_version": "1.0", "z_rel": None})
    write_json(manifest, {"schema_version": "1.0", "path": "redacted"})
    audit = audit_skeleton_artifacts(tmp_path, video, [quality, manifest])
    assert audit["passed"] is True
    assert audit["checks"]["video_has_no_audio"] is True
