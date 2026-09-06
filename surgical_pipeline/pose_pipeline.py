"""Opt-in skeleton-only pose pilot; it is intentionally separate from the normal video pipeline."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import threading
import time

import pandas as pd

from .analytics import build_instrument_summary, confirmed_health_count, health_statistics
from .config import AppConfig
from .depth import DepthEstimator
from .detections import extract_detections, filter_class
from .model_loader import load_validated_yolo_models, resolve_model_file_paths
from .pipeline import AnalysisCancelled, MaskTrackFallback, _device
from .pose_estimator import PoseEstimator
from .pose_schemas import PosePoint4D, PoseQuality
from .pose_tracker import PoseTracker, match_health_personnel
from .privacy_audit import audit_skeleton_artifacts
from .skeleton_renderer import render_skeleton_frame
from .spatiotemporal_tracking import instrument_point_4d, pose_points_4d, smooth_pose_points
from .trackers import tracked_result, write_tracker_config
from .utils import configure_logging, scrub_filename, sha256_file, write_json
from .video_io import VideoReader, VideoWriter, ensure_output_space


@dataclass(frozen=True)
class PosePilotResult:
    run_dir: Path
    skeleton_video: Path
    quality_report: Path
    privacy_report: Path
    manifest: Path
    files: list[Path]


PoseProgressCallback = Callable[[int, int, float, str], None]


def _pilot_run_dir(output_root: Path) -> Path:
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / ("pose_pilot_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S"))
    suffix = 1
    while candidate.exists():
        candidate = root / ("pose_pilot_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S") + f"_{suffix}")
        suffix += 1
    candidate.mkdir()
    return candidate


def _write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> Path:
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def _trajectory_html(points: list[PosePoint4D], path: Path) -> Path:
    """Write a geometry-only trajectory view; it never embeds images or video."""
    import plotly.graph_objects as go

    figure = go.Figure()
    grouped: dict[tuple[int, int], list[PosePoint4D]] = defaultdict(list)
    for point in points:
        if point.depth_valid:
            grouped[(point.pose_track_id, point.keypoint_index)].append(point)
    for (track_id, keypoint_index), items in grouped.items():
        items.sort(key=lambda value: value.timestamp_s)
        figure.add_trace(go.Scatter3d(x=[item.x_rel for item in items], y=[item.y_rel for item in items], z=[item.z_rel for item in items], mode="lines", name=f"P{track_id:02d} k{keypoint_index}"))
    figure.update_layout(title="Relative 4D trajectories (x_rel, y_rel, z_rel, t)", scene={"xaxis_title": "x_rel", "yaxis_title": "y_rel", "zaxis_title": "z_rel (monocular relative depth)"})
    figure.write_html(path, include_plotlyjs="inline", full_html=True)
    return path


def run_pose_pilot(
    input_path: Path | str,
    health_model_path: Path | str,
    instrument_model_path: Path | str,
    output_dir: Path | str,
    pose_model_path: Path | str = "yolo11m-pose.pt",
    config: AppConfig | None = None,
    max_seconds: float = 60.0,
    keypoint_confidence: float = 0.25,
    progress: PoseProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> PosePilotResult:
    """Run a 30–60 second synthetic-only pose assessment without copying source media."""
    input_video = Path(input_path).expanduser().resolve()
    if not input_video.is_file():
        raise FileNotFoundError("Pose pilot girdi videosu bulunamadı.")
    if not 0 < max_seconds <= 60:
        raise ValueError("Pose pilot süresi 0 ile 60 saniye arasında olmalıdır.")
    active_config = (config or AppConfig()).validate()
    run_dir = _pilot_run_dir(Path(output_dir))
    logger = configure_logging(run_dir / "pipeline.log", active_config.log_level)
    health_path, instrument_path = resolve_model_file_paths(Path.cwd(), Path(health_model_path), Path(instrument_model_path))
    device, use_half = _device(active_config.device, active_config.fp16)
    health_model, instrument_model, health_info, instrument_info, health_id = load_validated_yolo_models(
        health_path, instrument_path, active_config.health_class_name, active_config.health_class_id,
    )
    pose_estimator = PoseEstimator(pose_model_path, device, active_config.fp16, keypoint_confidence)
    pose_info = pose_estimator.load()
    reader = VideoReader(input_video)
    ensure_output_space(run_dir, input_video.stat().st_size, reader.metadata)
    writer = VideoWriter(run_dir / "skeleton_tracking_silent.mp4", reader.metadata, active_config.output_codec)
    health_tracker_file = write_tracker_config(run_dir / "health_tracker.yaml", active_config, active_config.health_confidence, active_config.health_iou)
    instrument_tracker_file = write_tracker_config(run_dir / "instrument_tracker.yaml", active_config, active_config.instrument_confidence, active_config.instrument_iou)
    health_fallback = MaskTrackFallback(active_config.track_buffer)
    instrument_fallback = MaskTrackFallback(active_config.track_buffer)
    pose_tracker = PoseTracker(max_lost_frames=max(1, int(reader.metadata.fps * active_config.max_lost_seconds)))
    depth = DepthEstimator(active_config.depth_model_id, active_config.depth_device, active_config.depth_enabled)
    quality = PoseQuality()
    previous_centres: dict[int, tuple[float, float]] = {}
    health_history: dict[int, list[float]] = {}
    trajectories: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    pose_rows: list[dict] = []
    keypoint_points: list[PosePoint4D] = []
    instrument_rows: list[dict] = []
    all_instruments = []
    health_count_rows: list[dict[str, float | int]] = []
    processed = 0
    started = time.perf_counter()
    manifest_path = run_dir / "run_manifest.json"
    try:
        for frame_index, timestamp_s, frame in reader:
            if cancel_event and cancel_event.is_set():
                raise AnalysisCancelled("Pose analizi kullanıcı tarafından iptal edildi.")
            if timestamp_s >= max_seconds:
                break
            health_result = tracked_result(health_model, frame, health_tracker_file, active_config.health_confidence, active_config.health_iou, device, use_half, active_config.image_size)
            instrument_result = tracked_result(instrument_model, frame, instrument_tracker_file, active_config.instrument_confidence, active_config.instrument_iou, device, use_half, active_config.image_size)
            health = health_fallback.assign(filter_class(extract_detections(health_result, health_info.names, frame.shape[:2], frame_index, timestamp_s), health_id), frame_index)
            instruments = instrument_fallback.assign(extract_detections(instrument_result, instrument_info.names, frame.shape[:2], frame_index, timestamp_s), frame_index)
            all_instruments.extend(instruments)
            poses = pose_tracker.update(match_health_personnel(pose_estimator.estimate(frame, frame_index, timestamp_s), health, keypoint_confidence), frame_index, keypoint_confidence)
            active_ids = {item.track_id for item in health if item.track_id is not None}
            personnel_count = confirmed_health_count(health_history, active_ids, timestamp_s, active_config.min_confirm_frames, active_config.max_lost_seconds)
            health_count_rows.append({"frame_index": frame_index, "timestamp_s": timestamp_s, "active_health_person_count": personnel_count})
            depth_map = None
            if active_config.depth_enabled and (poses or instruments):
                depth_map, _ = depth.cached_or_estimate(frame, frame_index, active_config.depth_stride)
            for pose in poses:
                pose_rows.append(pose.row_2d())
                keypoint_points.extend(pose_points_4d(pose, reader.metadata.width, reader.metadata.height, depth_map, pose_info.keypoint_names, keypoint_confidence))
            for instrument in instruments:
                point = instrument_point_4d(instrument, reader.metadata.width, reader.metadata.height, depth_map)
                if point is not None:
                    instrument_rows.append(point)
                centre = instrument.centroid()
                if centre is not None:
                    trajectories[(instrument.class_name, instrument.track_id or -1)].append((int(round(centre[0])), int(round(centre[1]))))
            quality.add(poses, keypoint_confidence, previous_centres)
            writer.write(render_skeleton_frame((reader.metadata.width, reader.metadata.height), timestamp_s, poses, instruments, personnel_count, trajectories, keypoint_confidence))
            processed += 1
            if progress and (processed == 1 or processed % 5 == 0 or processed == reader.metadata.frame_count):
                elapsed = time.perf_counter() - started
                progress(processed, reader.metadata.frame_count, (elapsed / processed) * max(reader.metadata.frame_count - processed, 0), "İskelet kareleri işleniyor")
    except AnalysisCancelled as error:
        logger.warning("Pose pilot cancelled: %s", error)
        writer.abort()
        write_json(manifest_path, {"schema_version": "1.0", "status": "cancelled", "run_id": run_dir.name, "processed_frame_count": processed, "audio_copied": False, "error": str(error)})
        raise
    except Exception:
        logger.exception("Pose pilot failed.")
        writer.abort()
        raise
    finally:
        reader.close()
    writer.close()
    warnings = writer.finalise(input_video, run_dir / "skeleton_tracking.mp4", keep_audio=False)
    processed_duration = processed / reader.metadata.fps
    instrument_visibility, visible_intervals = build_instrument_summary(
        all_instruments,
        [],
        active_config.max_gap_seconds,
        1.0 / reader.metadata.fps,
        active_config.max_relative_jump,
        active_config.minimum_interval_seconds,
        processed_duration,
        instrument_info.names.values(),
    )
    visibility_path = run_dir / "instrument_visibility_summary.json"
    write_json(
        visibility_path,
        {
            "schema_version": "1.0",
            "processed_duration_seconds": processed_duration,
            "processed_frame_count": processed,
            "fps": reader.metadata.fps,
            "health_person_statistics": health_statistics(health_count_rows),
            "instruments": instrument_visibility,
        },
    )
    visible_path = _write_csv(
        run_dir / "usage_intervals.csv",
        [{"schema_version": "1.0", **item.row()} for item in visible_intervals],
        ["schema_version", "class_name", "track_id", "start_s", "end_s", "duration_s", "source"],
    )
    absent_path = _write_csv(
        run_dir / "not_visible_intervals.csv",
        [
            {"schema_version": "1.0", "class_name": class_name, **interval, "source": "visible-table-complement"}
            for class_name, values in sorted(instrument_visibility.items())
            for interval in values.get("not_visible_intervals", [])
        ],
        ["schema_version", "class_name", "start_s", "end_s", "duration_s", "source"],
    )
    health_counts_path = _write_csv(
        run_dir / "health_person_count.csv",
        [{"schema_version": "1.0", **row} for row in health_count_rows],
        ["schema_version", "frame_index", "timestamp_s", "active_health_person_count"],
    )
    quality.track_loss_count = pose_tracker.short_loss_count
    quality.id_switch_estimate = pose_tracker.id_switch_estimate
    keypoint_points = smooth_pose_points(keypoint_points, active_config.smoothing_window)
    keypoint_rows = [point.row() for point in keypoint_points]
    quality_payload = quality.report(pose_tracker.track_lengths)
    quality_payload.update({"pose_model": pose_info.safe_dict(), "keypoint_confidence_threshold": keypoint_confidence, "processed_frame_count": processed, "warnings": warnings, "depth_enabled": active_config.depth_enabled})
    quality_path = run_dir / "pose_quality_report.json"
    write_json(quality_path, quality_payload)
    tracks_path = _write_csv(run_dir / "pose_tracks_2d.csv", pose_rows)
    keypoints_path = _write_csv(run_dir / "pose_keypoints_3d.csv", keypoint_rows)
    instruments_path = _write_csv(run_dir / "instrument_tracks_4d.csv", instrument_rows)
    manifest = {
        "schema_version": "1.0", "status": "success", "run_id": run_dir.name,
        "source_video": {"filename": scrub_filename(input_video), "sha256": sha256_file(input_video)},
        "models": {"health": health_info.safe_dict(), "instrument": instrument_info.safe_dict(), "pose": pose_info.safe_dict(), "health_class_id": health_id},
        "processed_frame_count": processed, "max_seconds": max_seconds, "audio_copied": False,
        "instrument_visibility_summary": visibility_path.name,
        "relative_4d_definition": "(x_rel, y_rel, z_rel, t): image-centre-relative x/y, normalized monocular z, and video timestamp. It is not metric 4D reconstruction.",
        "depth_status": "enabled_relative" if active_config.depth_enabled else "not_produced",
        "smoothing": {"method": "causal_rolling_median", "window": active_config.smoothing_window},
        "warnings": warnings,
    }
    write_json(manifest_path, manifest)
    extra_files: list[Path] = []
    if active_config.depth_enabled and any(row.get("depth_valid") for row in keypoint_rows):
        extra_files.append(_trajectory_html(keypoint_points, run_dir / "relative_4d_trajectories.html"))
    privacy_path = run_dir / "privacy_report.json"
    privacy = audit_skeleton_artifacts(run_dir, run_dir / "skeleton_tracking.mp4", [quality_path, manifest_path])
    write_json(privacy_path, privacy)
    logger.info("Pose pilot complete: %s frames; source RGB and audio were not copied.", processed)
    files = [run_dir / "skeleton_tracking.mp4", tracks_path, keypoints_path, instruments_path, health_counts_path, visible_path, absent_path, visibility_path, quality_path, privacy_path, manifest_path, run_dir / "pipeline.log", *extra_files]
    return PosePilotResult(run_dir, run_dir / "skeleton_tracking.mp4", quality_path, privacy_path, manifest_path, files)
