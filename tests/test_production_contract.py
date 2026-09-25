import json
import csv
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from surgical_pipeline.analytics import usage_intervals
from surgical_pipeline.config import AppConfig, load_config
from surgical_pipeline.coordinates_3d import point_from_detection
from surgical_pipeline.model_loader import ModelValidationError, validate_model_roles
from surgical_pipeline.pipeline import AnalysisCancelled, AnalysisPipeline
from surgical_pipeline.privacy_blur import PrivacyMasker
from surgical_pipeline.schemas import Detection, ModelInfo
from surgical_pipeline.utils import write_json


def _detection(timestamp: float, track_id: int = 1) -> Detection:
    return Detection(0, "scissors", 0.8, track_id, np.ones((5, 5), dtype=np.uint8), int(timestamp * 10), timestamp)


def test_model_roles_reject_reused_weights_and_accept_distinct_segment_models(tmp_path: Path) -> None:
    health_path, instrument_path = tmp_path / "health.pt", tmp_path / "instrument.pt"
    health = ModelInfo(str(health_path), "health-hash", {2: "monitor", 7: "health_personnel"}, "segment")
    instrument = ModelInfo(str(instrument_path), "instrument-hash", {0: "scissors"}, "segment")
    assert validate_model_roles(health_path, instrument_path, health, instrument) == 7
    with pytest.raises(ModelValidationError, match="Aynı model"):
        validate_model_roles(health_path, health_path, health, health)


def test_bbox_fallback_blurs_only_confident_health_bbox() -> None:
    frame = np.zeros((30, 30, 3), dtype=np.uint8)
    frame[8:22, 8:22] = 255
    empty_mask = np.zeros((30, 30), dtype=np.uint8)
    detection = Detection(0, "health_personel", 0.9, None, empty_mask, 0, 0.0, (9, 9, 21, 21), "missing_mask")
    result = PrivacyMasker(blur_kernel=9, dilation_px=0, feather_px=0, bbox_fallback=True).apply_with_quality(frame, [detection], 0)
    assert result.bbox_fallback_instances == 1
    assert np.array_equal(result.frame[0, 0], frame[0, 0])
    assert not np.array_equal(result.frame[9, 9], frame[9, 9])


def test_minimum_interval_filters_one_frame_false_positive() -> None:
    assert usage_intervals([_detection(0.0)], max_gap_seconds=0.5, frame_step_s=0.1, minimum_interval_seconds=0.2) == []


def test_relative_point_uses_bbox_and_centre_depth_fallback() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    detection = Detection(0, "scissors", 0.8, 2, mask, 0, 0.0, (3, 3, 7, 7), "missing_mask")
    depth = np.full((10, 10), 0.6, dtype=np.float32)
    point = point_from_detection(detection, depth, 10, 10)
    assert point is not None
    assert point.centroid_source == "bbox_center"
    assert point.depth_source == "centre_patch_fallback"
    assert point.depth_valid is True
    assert point.relative_x == pytest.approx(1 / 9)


def test_json_writer_removes_non_standard_float_values(tmp_path: Path) -> None:
    path = tmp_path / "strict.json"
    write_json(path, {"nan": float("nan"), "inf": float("inf")})
    assert json.loads(path.read_text(encoding="utf-8")) == {"nan": None, "inf": None}


def _synthetic_source(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (48, 36))
    for index in range(4):
        writer.write(np.full((36, 48, 3), 30 + index * 20, dtype=np.uint8))
    writer.release()


def _fake_detector(role: str, frame: np.ndarray, index: int, timestamp: float) -> list[Detection]:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if role == "health":
        mask[3:16, 3:15] = 1
        return [Detection(0, "health_personel", 0.9, 10, mask, index, timestamp)]
    mask[19:28, 18 + index : 28 + index] = 1
    return [Detection(0, "scissors", 0.8, 20, mask, index, timestamp)]


def test_mock_pipeline_writes_versioned_contract(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _synthetic_source(source)
    config = AppConfig(output_dir=tmp_path / "out", depth_enabled=False, min_confirm_frames=1, blur_kernel=9).validate()
    result = AnalysisPipeline(tmp_path, config, _fake_detector).run(source)
    expected = {"processed_video.mp4", "detections.csv", "tracks.csv", "usage_summary.csv", "usage_intervals.csv", "trajectories_3d.csv", "analysis.json", "run_manifest.json", "quality_report.json", "pipeline.log"}
    assert expected <= {path.name for path in result.run_dir.iterdir()}
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["models"]["health"]["safe_path"] == "mock.pt"
    assert "NaN" not in result.summary.read_text(encoding="utf-8")
    header = (result.run_dir / "tracks.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("schema_version,frame_index,timestamp_s,track_id")


def test_detector_runner_can_use_alternative_class_names(tmp_path: Path) -> None:
    from dataclasses import replace

    source = tmp_path / "alternative.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 24))
    for _ in range(2):
        writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()

    def detector(role: str, frame: np.ndarray, frame_index: int, timestamp: float) -> list[Detection]:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[5:12, 6:14] = 1
        name = "operator" if role == "health" else "needle"
        return [Detection(0, name, 0.9, 1, mask, frame_index, timestamp, (6, 5, 14, 12))]

    config = replace(load_config(tmp_path), output_dir=tmp_path / "alt-output", depth_enabled=False, health_class_name="operator", instrument_display_names={"needle": "needle"})
    result = AnalysisPipeline(tmp_path, config, detector).run(source)
    summary = json.loads((result.run_dir / "analysis.json").read_text(encoding="utf-8"))
    assert summary["models"]["health_person_class_name"] == "operator"
    with (result.run_dir / "detections.csv").open(encoding="utf-8", newline="") as handle:
        assert any(row["class_name"] == "needle" for row in csv.DictReader(handle))


def test_cancelled_mock_pipeline_has_manifest_but_no_final_video(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _synthetic_source(source)
    event = threading.Event()
    event.set()
    config = AppConfig(output_dir=tmp_path / "out", depth_enabled=False, blur_kernel=9).validate()
    with pytest.raises(AnalysisCancelled):
        AnalysisPipeline(tmp_path, config, _fake_detector).run(source, cancel_event=event)
    run_dir = next((tmp_path / "out").glob("run_*"))
    assert json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))["status"] == "cancelled"
    assert not (run_dir / "processed_video.mp4").exists()
