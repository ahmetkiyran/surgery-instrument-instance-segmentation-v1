"""End-to-end, local-only orchestration for surgical video analysis."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

import cv2
import numpy as np

from .analytics import build_instrument_summary, confirmed_health_count, health_statistics
from .charts import health_count_chart, relative_motion_chart, trajectories_charts, usage_duration_chart
from .config import AppConfig, load_config, write_run_config
from .coordinates_3d import point_from_detection, smooth_points
from .depth import DepthEstimator
from .detections import extract_detections, filter_class
from .model_loader import inspect_model, load_yolo_models, resolve_model_paths
from .privacy_blur import PrivacyMasker
from .report import create_results_zip, write_csvs, write_html_report, write_summary
from .schemas import AnalysisState, Detection, ModelInfo, SummarySchema
from .trackers import tracked_result, write_tracker_config
from .utils import configure_logging, hardware_summary, safe_run_dir
from .video_io import VideoReader, VideoWriter


class AnalysisCancelled(RuntimeError):
    pass


ProgressCallback = Callable[[int, int, float, str], None]
DetectorRunner = Callable[[str, np.ndarray, int, float], list[Detection]]


@dataclass(frozen=True)
class AnalysisResult:
    run_dir: Path
    processed_video: Path
    summary: Path
    files: list[Path]


class MaskTrackFallback:
    """Last-resort IoU association for a detector result lacking tracker IDs.

    Ultralytics BoT-SORT/ByteTrack remains the primary tracker. This only retains
    privacy and analytics safety if a tracker does not emit an ID on an isolated frame.
    """

    def __init__(self, max_age_frames: int = 30) -> None:
        self.max_age_frames = max_age_frames
        self.next_id = 1
        self.tracks: dict[int, tuple[str, np.ndarray, int]] = {}

    @staticmethod
    def _iou(left: np.ndarray, right: np.ndarray) -> float:
        intersection = np.logical_and(left > 0, right > 0).sum()
        union = np.logical_or(left > 0, right > 0).sum()
        return float(intersection / union) if union else 0.0

    def assign(self, detections: list[Detection], frame_index: int) -> list[Detection]:
        for track_id, (_, _, last_seen) in list(self.tracks.items()):
            if frame_index - last_seen > self.max_age_frames:
                del self.tracks[track_id]
        for detection in detections:
            if detection.track_id is not None:
                self.tracks[detection.track_id] = (detection.class_name, detection.mask, frame_index)
                continue
            matches = [(self._iou(detection.mask, mask), track_id) for track_id, (name, mask, _) in self.tracks.items() if name == detection.class_name]
            score, track_id = max(matches, default=(0.0, -1))
            if score < 0.3:
                track_id = self.next_id; self.next_id += 1
            detection.track_id = track_id
            self.tracks[track_id] = (detection.class_name, detection.mask, frame_index)
        return detections


def _device(configured: str) -> tuple[str | int, bool]:
    hardware = hardware_summary()
    if configured == "auto":
        return (0, True) if hardware["cuda_available"] else ("cpu", False)
    is_cuda = str(configured).lower() not in {"cpu", "mps"}
    return configured, is_cuda


def _draw_text(frame: np.ndarray, text: str, xy: tuple[int, int], size: int = 18) -> None:
    """Use a Unicode-capable font when available, falling back to OpenCV."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        font_path = Path("C:/Windows/Fonts/arial.ttf")
        font = ImageFont.truetype(str(font_path), size) if font_path.is_file() else ImageFont.load_default()
        box = draw.textbbox(xy, text, font=font)
        draw.rounded_rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), radius=3, fill=(2, 12, 24))
        draw.text(xy, text, fill=(218, 255, 255), font=font)
        frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(frame, text.encode("ascii", "replace").decode(), xy, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 255, 255), 2, cv2.LINE_AA)


def _draw_instrument_labels(frame: np.ndarray, detections: list[Detection], display_names: dict[str, str]) -> None:
    occupied: list[tuple[int, int, int, int]] = []
    height, width = frame.shape[:2]
    for detection in sorted(detections, key=lambda item: int(np.argwhere(item.mask > 0)[:, 0].min()) if np.any(item.mask) else height):
        ys, xs = np.where(detection.mask > 0)
        if not len(xs):
            continue
        label = display_names.get(detection.class_name, detection.class_name)
        x = int(np.median(xs)); y = max(20, int(ys.min()) - 6)
        estimate_width = max(70, len(label) * 9)
        for _ in range(12):
            candidate = (x, y - 18, min(width, x + estimate_width), y + 5)
            if not any(candidate[0] < other[2] and candidate[2] > other[0] and candidate[1] < other[3] and candidate[3] > other[1] for other in occupied):
                break
            y = min(height - 8, y + 22)
        occupied.append((x, y - 18, min(width, x + estimate_width), y + 5))
        _draw_text(frame, label, (x, y), 17)


class AnalysisPipeline:
    def __init__(self, project_root: Path | None = None, config: AppConfig | None = None, detector_runner: DetectorRunner | None = None) -> None:
        self.project_root = (project_root or Path.cwd()).resolve()
        self.config = config or load_config(self.project_root)
        self.detector_runner = detector_runner

    def preflight(self, health_model: Path | None = None, instrument_model: Path | None = None) -> tuple[Path, Path, int]:
        health, instrument, health_id, _ = resolve_model_paths(self.project_root, health_model or self.config.health_model_path, instrument_model or self.config.instrument_model_path)
        return health, instrument, health_id

    def run(self, input_video: Path, health_model: Path | None = None, instrument_model: Path | None = None, progress: ProgressCallback | None = None, cancel_event: threading.Event | None = None) -> AnalysisResult:
        input_video = input_video.expanduser().resolve()
        if not input_video.is_file():
            raise FileNotFoundError("Girdi videosu bulunamadı.")
        if self.detector_runner:
            # Dependency-injected detectors are intentionally limited to tests and do not bypass
            # the real-model preflight in normal CLI/UI operation.
            health_path = instrument_path = Path("mock.pt")
            health_class_id = 0
            health_info = ModelInfo(path="mock.pt", sha256="mock", names={0: "health_personel"}, task="segment")
            instrument_info = ModelInfo(path="mock.pt", sha256="mock", names={0: "scissors"}, task="segment")
        else:
            health_path, instrument_path, health_class_id = self.preflight(health_model, instrument_model)
            health_info, instrument_info = inspect_model(health_path), inspect_model(instrument_path)
        device, use_half = _device(self.config.device)
        output_root = self.config.output_dir if self.config.output_dir.is_absolute() else self.project_root / self.config.output_dir
        run_dir = safe_run_dir(output_root)
        logger = configure_logging(run_dir / "logs" / "analysis.log")
        logger.info("Analysis started with private source file; source path will not be persisted in reports.")
        write_run_config(run_dir / "run_config.yaml", self.config)
        if self.detector_runner:
            health_yolo = instrument_yolo = None
        else:
            health_yolo, instrument_yolo = load_yolo_models(health_path, instrument_path)
        health_tracker_path = write_tracker_config(
            run_dir / "health_tracker.yaml",
            self.config,
            self.config.health_confidence,
            self.config.health_iou,
        )
        instrument_tracker_path = write_tracker_config(
            run_dir / "instrument_tracker.yaml",
            self.config,
            self.config.instrument_confidence,
            self.config.instrument_iou,
        )
        reader = VideoReader(input_video)
        temporary_video = run_dir / "processed_video_silent.mp4"
        writer = VideoWriter(temporary_video, reader.metadata)
        privacy = PrivacyMasker(self.config.privacy_mode, self.config.blur_kernel, self.config.pixelation_factor, self.config.mask_dilation_px, self.config.feather_px, self.config.persistence_frames)
        depth = DepthEstimator(self.config.depth_model_id, self.config.depth_device, self.config.depth_enabled)
        state = AnalysisState()
        health_history: dict[int, list[float]] = {}
        health_fallback, instrument_fallback = MaskTrackFallback(self.config.track_buffer), MaskTrackFallback(self.config.track_buffer)
        all_instruments: list[Detection] = []
        started = time.perf_counter()
        processed = 0
        try:
            for frame_index, timestamp_s, frame in reader:
                if cancel_event and cancel_event.is_set():
                    raise AnalysisCancelled("Analiz kullanıcı tarafından iptal edildi.")
                if self.config.max_video_seconds is not None and timestamp_s > self.config.max_video_seconds:
                    break
                if self.detector_runner:
                    health_detections = self.detector_runner("health", frame, frame_index, timestamp_s)
                    instrument_detections = self.detector_runner("instrument", frame, frame_index, timestamp_s)
                else:
                    health_result = tracked_result(health_yolo, frame, health_tracker_path, self.config.health_confidence, self.config.health_iou, device, use_half)
                    instrument_result = tracked_result(instrument_yolo, frame, instrument_tracker_path, self.config.instrument_confidence, self.config.instrument_iou, device, use_half)
                    health_detections = filter_class(extract_detections(health_result, health_info.names, frame.shape[:2], frame_index, timestamp_s), health_class_id)
                    instrument_detections = extract_detections(instrument_result, instrument_info.names, frame.shape[:2], frame_index, timestamp_s)
                health_detections = health_fallback.assign(health_detections, frame_index)
                instrument_detections = instrument_fallback.assign(instrument_detections, frame_index)
                active_ids = {item.track_id for item in health_detections if item.track_id is not None}
                active_count = confirmed_health_count(health_history, active_ids, timestamp_s, self.config.min_confirm_frames, self.config.max_lost_seconds)
                state.health_counts.append({"frame_index": frame_index, "timestamp_s": timestamp_s, "active_health_person_count": active_count})
                anonymised = privacy.apply(frame, health_detections, frame_index)
                _draw_instrument_labels(anonymised, instrument_detections, self.config.instrument_display_names)
                _draw_text(anonymised, f"Sağlık personeli: {active_count}", (18, 24), 19)
                if instrument_detections:
                    depth_map = None
                    if self.config.depth_enabled:
                        depth_map, _ = depth.cached_or_estimate(frame, frame_index, self.config.depth_stride)
                    for detection in instrument_detections:
                        point = point_from_detection(detection, depth_map, reader.metadata.width, reader.metadata.height)
                        if point:
                            state.track_points.append(point)
                all_instruments.extend(instrument_detections)
                writer.write(anonymised)
                processed += 1
                if progress and (processed == 1 or processed % 5 == 0 or processed == reader.metadata.frame_count):
                    elapsed = time.perf_counter() - started
                    eta = (elapsed / processed) * max(reader.metadata.frame_count - processed, 0)
                    progress(processed, reader.metadata.frame_count, eta, "Kareler işleniyor")
        except Exception:
            logger.exception("Analysis interrupted by an exception.")
            raise
        finally:
            reader.close(); writer.close()
        state.track_points = smooth_points(state.track_points, self.config.smoothing_window)
        frame_step = 1.0 / reader.metadata.fps
        instrument_summary, intervals = build_instrument_summary(all_instruments, state.track_points, self.config.max_gap_seconds, frame_step, self.config.max_relative_jump)
        elapsed = time.perf_counter() - started
        warnings = writer.finalise(input_video, run_dir / "processed_video.mp4", self.config.save_audio)
        if not self.config.depth_enabled:
            warnings.append("Göreli 3B tracking kullanıcı ayarıyla kapatıldı.")
        if reader.variable_timestamps:
            warnings.append("Değişken kare zamanları algılandı; OpenCV'nin mevcut zaman damgaları kullanıldı.")
        state.warnings.extend(warnings)
        summary_data = {
            "schema_version": "1.0",
            "video": {**reader.metadata.safe_dict(), "processed_frame_count": processed, "duration_seconds_processed": processed / reader.metadata.fps, "source_filename": "redacted"},
            "models": {"health": health_info.safe_dict(), "instrument": instrument_info.safe_dict(), "health_person_class_id": health_class_id},
            "health_person_statistics": health_statistics(state.health_counts),
            "instrument_usage": instrument_summary,
            "relative_3d_note": "Relative 3D trajectory. Values are relative motion units, not calibrated real-world distances.",
            "processing": {"elapsed_seconds": elapsed, "average_processing_fps": processed / elapsed if elapsed else 0.0, "hardware": hardware_summary(), "mode": "GPU FP16" if use_half else "CPU FP32"},
            "warnings": state.warnings,
        }
        SummarySchema.model_validate(summary_data)
        files = list(write_csvs(run_dir, state.track_points, state.health_counts, intervals))
        files += [usage_duration_chart(instrument_summary, run_dir / "usage_duration_chart.png"), relative_motion_chart(instrument_summary, run_dir / "relative_3d_motion_chart.png"), health_count_chart(state.health_counts, run_dir / "health_person_count_chart.png")]
        files += list(trajectories_charts(state.track_points, run_dir / "trajectories_3d.html", run_dir / "trajectories_3d.png"))
        summary_path = write_summary(run_dir, summary_data); report_path = write_html_report(run_dir, summary_data)
        zip_path = create_results_zip(run_dir)
        files += [run_dir / "processed_video.mp4", summary_path, report_path, run_dir / "run_config.yaml", zip_path]
        logger.info("Analysis complete: %s frames in %.2fs", processed, elapsed)
        return AnalysisResult(run_dir, run_dir / "processed_video.mp4", summary_path, files)
