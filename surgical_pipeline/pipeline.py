"""UI-independent orchestration for privacy-first surgical video analysis."""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .analytics import build_instrument_summary, confirmed_health_count, health_statistics
from .charts import health_count_chart, relative_motion_chart, trajectories_charts, usage_duration_chart
from .config import AppConfig, load_config, write_run_config
from .coordinates_3d import point_from_detection, smooth_points
from .depth import DepthEstimator
from .detections import extract_detections, filter_class
from .model_loader import load_validated_yolo_models, resolve_model_file_paths
from .privacy_blur import PrivacyMasker
from .report import create_results_zip, write_csvs, write_html_report, write_summary
from .schemas import AnalysisState, Detection, ModelInfo, QualityReportSchema, RunManifestSchema, SummarySchema
from .trackers import tracked_result, write_tracker_config
from .utils import configure_logging, hardware_summary, safe_run_dir, scrub_filename, sha256_file, write_json
from .video_io import VideoReader, VideoWriter, ensure_output_space


class AnalysisCancelled(RuntimeError):
    """A caller requested cancellation at a safe frame boundary."""


ProgressCallback = Callable[[int, int, float, str], None]
DetectorRunner = Callable[[str, np.ndarray, int, float], list[Detection]]


@dataclass(frozen=True)
class AnalysisResult:
    """Structured public result of a completed, validated analysis run."""

    run_dir: Path
    processed_video: Path
    summary: Path
    manifest: Path
    quality_report: Path
    status: str
    files: list[Path]


class MaskTrackFallback:
    """Small IoU adapter used only when the selected tracker omits an ID."""

    def __init__(self, max_age_frames: int = 30) -> None:
        self.max_age_frames = max_age_frames
        self.next_id = 1
        self.tracks: dict[int, tuple[str, np.ndarray, tuple[float, float, float, float] | None, int]] = {}

    @staticmethod
    def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
        intersection = np.logical_and(left > 0, right > 0).sum()
        union = np.logical_or(left > 0, right > 0).sum()
        return float(intersection / union) if union else 0.0

    @staticmethod
    def _bbox_iou(left: tuple[float, float, float, float] | None, right: tuple[float, float, float, float] | None) -> float:
        if left is None or right is None:
            return 0.0
        x1, y1 = max(left[0], right[0]), max(left[1], right[1])
        x2, y2 = min(left[2], right[2]), min(left[3], right[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if not intersection:
            return 0.0
        left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        return intersection / max(left_area + right_area - intersection, 1e-9)

    def assign(self, detections: list[Detection], frame_index: int) -> list[Detection]:
        for track_id, (_, _, _, last_seen) in list(self.tracks.items()):
            if frame_index - last_seen > self.max_age_frames:
                del self.tracks[track_id]
        for detection in detections:
            if detection.track_id is not None:
                detection.tracking_state = "model_track"
                self.tracks[detection.track_id] = (detection.class_name, detection.mask, detection.bbox_xyxy, frame_index)
                continue
            matches = []
            for track_id, (name, mask, bbox, _) in self.tracks.items():
                if name != detection.class_name:
                    continue
                score = self._mask_iou(detection.mask, mask) if np.any(detection.mask) and np.any(mask) else self._bbox_iou(detection.bbox_xyxy, bbox)
                matches.append((score, track_id))
            score, track_id = max(matches, default=(0.0, -1))
            if score < 0.3:
                track_id = self.next_id
                self.next_id += 1
            detection.track_id = track_id
            detection.tracking_state = "fallback_iou"
            self.tracks[track_id] = (detection.class_name, detection.mask, detection.bbox_xyxy, frame_index)
        return detections


def _device(configured: str, fp16_preference: bool) -> tuple[str | int, bool]:
    hardware = hardware_summary()
    if configured == "auto":
        return (0, fp16_preference) if hardware["cuda_available"] else ("cpu", False)
    is_cuda = str(configured).lower().startswith("cuda") or str(configured) == "0"
    if is_cuda and not hardware["cuda_available"]:
        raise RuntimeError("CUDA cihazı istendi ancak kullanılabilir CUDA aygıtı yok.")
    return configured, bool(is_cuda and fp16_preference)


def _draw_text(frame: np.ndarray, text: str, xy: tuple[int, int], size: int = 18) -> None:
    """Draw presentation-only text; never draws IDs, boxes, masks, or confidences."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=size)
        box = draw.textbbox(xy, text, font=font)
        draw.rounded_rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), radius=3, fill=(2, 12, 24))
        draw.text(xy, text, fill=(218, 255, 255), font=font)
        frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(frame, text.encode("ascii", "replace").decode(), xy, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 255, 255), 2, cv2.LINE_AA)


def _draw_instrument_labels(frame: np.ndarray, detections: list[Detection], display_names: dict[str, str]) -> None:
    """Place only class names and keep labels inside the frame where practical."""
    occupied: list[tuple[int, int, int, int]] = []
    height, width = frame.shape[:2]
    for detection in sorted(detections, key=lambda item: int(np.argwhere(item.mask > 0)[:, 0].min()) if np.any(item.mask) else height):
        ys, xs = np.where(detection.mask > 0)
        if len(xs):
            x, y = int(np.median(xs)), max(20, int(ys.min()) - 6)
        elif detection.bbox_xyxy is not None:
            x1, y1, x2, _ = detection.bbox_xyxy
            x, y = int((x1 + x2) / 2), max(20, int(y1) - 6)
        else:
            continue
        label = display_names.get(detection.class_name, detection.class_name)
        x = min(max(0, x), max(0, width - 1))
        estimate_width = max(70, len(label) * 9)
        for _ in range(12):
            candidate = (x, max(0, y - 18), min(width, x + estimate_width), min(height, y + 5))
            if not any(candidate[0] < other[2] and candidate[2] > other[0] and candidate[1] < other[3] and candidate[3] > other[1] for other in occupied):
                break
            y = min(height - 8, y + 22)
        occupied.append((x, max(0, y - 18), min(width, x + estimate_width), min(height, y + 5)))
        _draw_text(frame, label, (x, y), 17)


class AnalysisPipeline:
    def __init__(self, project_root: Path | None = None, config: AppConfig | None = None, detector_runner: DetectorRunner | None = None) -> None:
        self.project_root = (project_root or Path.cwd()).resolve()
        self.config = (config or load_config(self.project_root)).validate()
        self.detector_runner = detector_runner

    def preflight(self, health_model: Path | None = None, instrument_model: Path | None = None) -> tuple[Path, Path]:
        return resolve_model_file_paths(
            self.project_root,
            health_model or self.config.health_model_path,
            instrument_model or self.config.instrument_model_path,
        )

    def run(self, input_video: Path, health_model: Path | None = None, instrument_model: Path | None = None, progress: ProgressCallback | None = None, cancel_event: threading.Event | None = None) -> AnalysisResult:
        """Run both models on originals, then anonymise the separate output copy."""
        input_video = input_video.expanduser().resolve()
        if not input_video.is_file():
            raise FileNotFoundError("Girdi videosu bulunamadı.")
        if self.detector_runner:
            health_path = instrument_path = Path("mock.pt")
            health_class_id = 0
            health_info = ModelInfo(path="mock.pt", sha256="mock", names={0: "health_personel"}, task="segment")
            instrument_info = ModelInfo(path="mock.pt", sha256="mock", names={0: "scissors"}, task="segment")
        else:
            health_path, instrument_path = self.preflight(health_model, instrument_model)
        device, use_half = _device(self.config.device, self.config.fp16)
        output_root = self.config.output_dir if self.config.output_dir.is_absolute() else self.project_root / self.config.output_dir
        run_dir = safe_run_dir(output_root)
        run_id = run_dir.name
        started_at = datetime.now(UTC)
        logger = configure_logging(run_dir / "pipeline.log", self.config.log_level)
        logger.info("Analysis started; private source path is not persisted in public artifacts.")
        write_run_config(run_dir / "run_config.yaml", self.config)
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
            try:
                import torch

                torch.manual_seed(self.config.random_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.config.random_seed)
            except ImportError:
                pass
        if self.detector_runner:
            health_yolo = instrument_yolo = None
        else:
            health_yolo, instrument_yolo, health_info, instrument_info, health_class_id = load_validated_yolo_models(
                health_path, instrument_path, self.config.health_class_name, self.config.health_class_id,
            )
        health_tracker_path = write_tracker_config(run_dir / "health_tracker.yaml", self.config, self.config.health_confidence, self.config.health_iou)
        instrument_tracker_path = write_tracker_config(run_dir / "instrument_tracker.yaml", self.config, self.config.instrument_confidence, self.config.instrument_iou)
        reader = VideoReader(input_video)
        ensure_output_space(run_dir, input_video.stat().st_size, reader.metadata)
        writer = VideoWriter(run_dir / "processed_video_silent.mp4", reader.metadata, self.config.output_codec)
        privacy = PrivacyMasker(self.config.privacy_mode, self.config.blur_kernel, self.config.pixelation_factor, self.config.mask_dilation_px, self.config.feather_px, self.config.persistence_frames, self.config.bbox_privacy_fallback)
        depth = DepthEstimator(self.config.depth_model_id, self.config.depth_device, self.config.depth_enabled)
        state = AnalysisState()
        health_history: dict[int, list[float]] = {}
        health_fallback = MaskTrackFallback(self.config.track_buffer)
        instrument_fallback = MaskTrackFallback(self.config.track_buffer)
        all_instruments: list[Detection] = []
        privacy_quality = {"mask_instances": 0, "bbox_fallback_instances": 0, "persisted_instances": 0}
        depth_quality = {"computed_frames": 0, "interpolated_frames": 0, "points_with_depth": 0, "points_missing_depth": 0}
        processed = 0
        started = time.perf_counter()
        manifest_path = run_dir / "run_manifest.json"
        quality_path = run_dir / "quality_report.json"

        def write_manifest(status: str, warnings: list[str] | None = None, error: str | None = None) -> None:
            payload = {
                "schema_version": "1.0", "pipeline_version": __version__, "run_id": run_id, "status": status,
                "started_at_utc": started_at.isoformat(), "finished_at_utc": datetime.now(UTC).isoformat(),
                "source_video": {"filename": scrub_filename(input_video), "size_bytes": input_video.stat().st_size, "sha256": sha256_file(input_video), **reader.metadata.safe_dict()},
                "models": {
                    "health": {**health_info.safe_dict(), "safe_path": Path(health_info.path).name},
                    "instrument": {**instrument_info.safe_dict(), "safe_path": Path(instrument_info.path).name},
                    "health_class": {"id": health_class_id, "name": health_info.names[health_class_id]},
                },
                "runtime": {"device": str(device), "fp16": use_half, "hardware": hardware_summary()},
                "config": self.config.public_run_config(),
                "timing": {"method": reader.timestamp_method, "variable_frame_rate_detected": reader.variable_timestamps},
                "depth_coordinate_system": "x/y are image-centre-relative normalized coordinates; z is normalized monocular depth. All 3D values are relative camera coordinates, never metric.",
                "scientific_limitations": ["Monocular depth is not metric depth.", "Relative 3D scales are not comparable across tracks or videos without calibration.", "Usage intervals describe visible model detections, not verified physical instrument use."],
                "processed_frame_count": processed, "written_frame_count": writer.frame_count, "warnings": warnings or [], "error": error,
            }
            RunManifestSchema.model_validate(payload)
            write_json(manifest_path, payload)

        try:
            for frame_index, timestamp_s, frame in reader:
                if cancel_event and cancel_event.is_set():
                    raise AnalysisCancelled("Analiz kullanıcı tarafından iptal edildi.")
                if self.config.max_video_seconds is not None and timestamp_s > self.config.max_video_seconds:
                    break
                if self.detector_runner:
                    health_detections = filter_class(self.detector_runner("health", frame, frame_index, timestamp_s), health_class_id)
                    instrument_detections = self.detector_runner("instrument", frame, frame_index, timestamp_s)
                else:
                    health_result = tracked_result(health_yolo, frame, health_tracker_path, self.config.health_confidence, self.config.health_iou, device, use_half, self.config.image_size)
                    instrument_result = tracked_result(instrument_yolo, frame, instrument_tracker_path, self.config.instrument_confidence, self.config.instrument_iou, device, use_half, self.config.image_size)
                    health_detections = filter_class(extract_detections(health_result, health_info.names, frame.shape[:2], frame_index, timestamp_s), health_class_id)
                    instrument_detections = extract_detections(instrument_result, instrument_info.names, frame.shape[:2], frame_index, timestamp_s)
                health_detections = health_fallback.assign(health_detections, frame_index)
                instrument_detections = instrument_fallback.assign(instrument_detections, frame_index)
                active_ids = {item.track_id for item in health_detections if item.track_id is not None}
                active_count = confirmed_health_count(health_history, active_ids, timestamp_s, self.config.min_confirm_frames, self.config.max_lost_seconds)
                state.health_counts.append({"frame_index": frame_index, "timestamp_s": timestamp_s, "active_health_person_count": active_count})

                depth_map = None
                depth_interpolated = False
                if instrument_detections and self.config.depth_enabled:
                    depth_map, computed = depth.cached_or_estimate(frame, frame_index, self.config.depth_stride)
                    depth_interpolated = not computed
                    depth_quality["computed_frames" if computed else "interpolated_frames"] += 1
                for detection in instrument_detections:
                    point = point_from_detection(detection, depth_map, reader.metadata.width, reader.metadata.height, depth_interpolated)
                    if point is not None:
                        state.track_points.append(point)
                        depth_quality["points_with_depth" if point.depth_valid else "points_missing_depth"] += 1

                # This is deliberately after inference: no analytics sees blurred pixels.
                privacy_result = privacy.apply_with_quality(frame.copy(), health_detections, frame_index)
                anonymised = privacy_result.frame
                privacy_quality["mask_instances"] += privacy_result.mask_instances
                privacy_quality["bbox_fallback_instances"] += privacy_result.bbox_fallback_instances
                privacy_quality["persisted_instances"] += privacy_result.persisted_instances
                _draw_instrument_labels(anonymised, instrument_detections, self.config.instrument_display_names)
                _draw_text(anonymised, f"Sağlık personeli: {active_count}", (18, 24), 19)
                all_instruments.extend(instrument_detections)
                writer.write(anonymised)
                processed += 1
                if progress and (processed == 1 or processed % 5 == 0 or processed == reader.metadata.frame_count):
                    elapsed = time.perf_counter() - started
                    eta = (elapsed / processed) * max(reader.metadata.frame_count - processed, 0)
                    progress(processed, reader.metadata.frame_count, eta, "Kareler işleniyor")

            state.track_points = smooth_points(state.track_points, self.config.smoothing_window)
            frame_step = 1.0 / reader.metadata.fps
            instrument_summary, intervals = build_instrument_summary(
                all_instruments,
                state.track_points,
                self.config.max_gap_seconds,
                frame_step,
                self.config.max_relative_jump,
                self.config.minimum_interval_seconds,
                processed / reader.metadata.fps,
                instrument_info.names.values(),
            )
            elapsed = time.perf_counter() - started
            writer.close()
            warnings = writer.finalise(input_video, run_dir / "processed_video.mp4", self.config.save_audio)
            if not self.config.depth_enabled:
                warnings.append("Göreli 3B tracking kullanıcı ayarıyla kapatıldı.")
            if reader.variable_timestamps:
                warnings.append("Değişken kare zamanları algılandı; kullanılabilir OpenCV PTS değerleri kullanıldı.")
            if privacy_quality["bbox_fallback_instances"]:
                warnings.append("Bazı health-personnel tespitlerinde maske yerine bbox privacy fallback kullanıldı.")
            state.warnings.extend(warnings)
            summary_data = {
                "schema_version": "1.0",
                "video": {**reader.metadata.safe_dict(), "processed_frame_count": processed, "written_frame_count": writer.frame_count, "duration_seconds_processed": processed / reader.metadata.fps, "source_filename": scrub_filename(input_video), "timestamp_method": reader.timestamp_method},
                "models": {"health": health_info.safe_dict(), "instrument": instrument_info.safe_dict(), "health_person_class_id": health_class_id, "health_person_class_name": health_info.names[health_class_id]},
                "health_person_statistics": health_statistics(state.health_counts), "instrument_usage": instrument_summary,
                "relative_3d_note": "x/y image-centre-relative and z normalized monocular-depth coordinates are relative camera coordinates, not calibrated real-world positions or metres.",
                "processing": {"elapsed_seconds": elapsed, "average_processing_fps": processed / elapsed if elapsed else 0.0, "hardware": hardware_summary(), "mode": "GPU FP16" if use_half else "FP32", "depth_smoothing": {"method": "causal_rolling_median", "window": self.config.smoothing_window}},
                "warnings": state.warnings,
            }
            SummarySchema.model_validate(summary_data)
            quality_data = {
                "schema_version": "1.0", "privacy": {**privacy_quality, "bbox_fallback_enabled": self.config.bbox_privacy_fallback},
                "depth": {**depth_quality, "enabled": self.config.depth_enabled, "stride": self.config.depth_stride, "interpolation": "last-valid-depth-map carry-forward; records are marked depth_interpolated"},
                "warnings": state.warnings,
            }
            QualityReportSchema.model_validate(quality_data)
            write_json(quality_path, quality_data)
            files = list(write_csvs(run_dir, all_instruments, state.track_points, state.health_counts, intervals, instrument_summary))
            files += [usage_duration_chart(instrument_summary, run_dir / "usage_duration_chart.png"), relative_motion_chart(instrument_summary, run_dir / "relative_3d_motion_chart.png"), health_count_chart(state.health_counts, run_dir / "health_person_count_chart.png")]
            files += list(trajectories_charts(state.track_points, run_dir / "trajectories_3d.html", run_dir / "trajectories_3d.png"))
            summary_path = write_summary(run_dir, summary_data)
            report_path = write_html_report(run_dir, summary_data)
            write_manifest("success", state.warnings)
            zip_path = create_results_zip(run_dir)
            files += [run_dir / "processed_video.mp4", summary_path, run_dir / "summary.json", manifest_path, quality_path, report_path, run_dir / "run_config.yaml", run_dir / "pipeline.log", zip_path]
            logger.info("Analysis complete: %s frames in %.2fs", processed, elapsed)
            return AnalysisResult(run_dir, run_dir / "processed_video.mp4", summary_path, manifest_path, quality_path, "success", files)
        except AnalysisCancelled as error:
            logger.warning("Analysis cancelled: %s", error)
            writer.abort()
            write_manifest("cancelled", state.warnings, str(error))
            raise
        except Exception as error:
            logger.exception("Analysis interrupted by an exception.")
            writer.abort()
            write_manifest("failed", state.warnings, str(error))
            if "out of memory" in str(error).casefold() and str(device).casefold() != "cpu":
                raise RuntimeError("CUDA belleği yetersiz. Görüntü boyutunu/depth aralığını azaltın veya --device cpu kullanın.") from error
            raise
        finally:
            reader.close()


def analyze_video(input_path: Path | str, health_model_path: Path | str, instrument_model_path: Path | str, output_dir: Path | str, config: AppConfig | None = None, progress_callback: ProgressCallback | None = None, cancellation_token: threading.Event | None = None) -> AnalysisResult:
    """Public UI-free API suitable for CLI, desktop, mobile, or service adapters."""
    base = config or load_config(Path.cwd())
    resolved_config = replace(base, output_dir=Path(output_dir)).validate()
    return AnalysisPipeline(Path.cwd(), resolved_config).run(Path(input_path), Path(health_model_path), Path(instrument_model_path), progress_callback, cancellation_token)
