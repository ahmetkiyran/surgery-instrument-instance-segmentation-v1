"""Shared, UI-independent analysis core for inspection and synthetic X-ray output.

This module deliberately keeps inference and rendering separate.  In particular,
``PrivacyXRayRenderer.render`` has no source-frame argument: it can only consume
geometry, masks, poses and metadata.  Front ends submit ``SelectionEvent`` values
to the same service/CLI contract; they never run a model themselves.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Literal
from uuid import uuid4

import cv2
import numpy as np

from .config import AppConfig
from .detections import extract_detections, filter_class
from .model_loader import load_validated_yolo_models, resolve_model_file_paths
from .pipeline import AnalysisCancelled, MaskTrackFallback, _device
from .pose_estimator import PoseEstimator
from .pose_schemas import COCO17_SKELETON
from .pose_tracker import PoseTracker, match_health_personnel
from .trackers import tracked_result, write_tracker_config
from .utils import sha256_file, write_json
from .video_io import VideoReader, VideoWriter

RenderMode = Literal["legacy", "inspection", "privacy-xray", "dual"]


@dataclass(slots=True)
class SelectionEvent:
    normalized_x: float
    normalized_y: float
    frame_index: int
    timestamp: float
    source_width: int
    source_height: int
    displayed_width: int | None = None
    displayed_height: int | None = None
    session_id: str | None = None
    target_category: str = "object"
    target_name: str | None = None
    action: str = "add"
    event_id: str = field(default_factory=lambda: str(uuid4()))
    sam3_track_id: int | None = None
    unified_track_id: str | None = None
    confidence: float | None = None
    created_frame_index: int | None = None
    initial_v1_track_id: int | None = None
    initial_sam3_track_id: int | None = None
    initial_class_name: str | None = None
    output_color: str | None = None

    def pixel(self, width: int, height: int) -> tuple[int, int]:
        """Map normalized, viewport-independent coordinates to the source frame."""
        return (
            min(width - 1, max(0, round(float(self.normalized_x) * (width - 1)))),
            min(height - 1, max(0, round(float(self.normalized_y) * (height - 1)))),
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SelectionEvent":
        allowed = {key: item for key, item in value.items() if key in cls.__dataclass_fields__}
        return cls(**allowed)


@dataclass(slots=True)
class UnifiedTrack:
    unified_track_id: str
    category: str
    v1_track_id: int | None = None
    sam3_track_id: int | None = None
    first_frame: int = 0
    last_frame: int = 0
    selected: bool = False
    recovered: bool = False
    id_switches: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class FrameAnalysis:
    frame_index: int
    timestamp: float
    frame_width: int
    frame_height: int
    persons: list[Any]
    instruments: list[Any]
    poses: list[Any]
    v1_tracks: list[int]
    sam3_tracks: list[dict[str, Any]]
    unified_tracks: list[UnifiedTrack]
    selected_tracks: list[str]
    active_selection_events: list[SelectionEvent]
    warnings: list[str] = field(default_factory=list)
    processing_status: str = "running"


def _bbox_iou(left: tuple[float, float, float, float] | None, right: tuple[float, float, float, float] | None) -> float:
    if left is None or right is None:
        return 0.0
    x1, y1, x2, y2 = max(left[0], right[0]), max(left[1], right[1]), min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(area_left + area_right - intersection, 1e-9)


def _mask_iou(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None or left.shape != right.shape:
        return 0.0
    union = np.count_nonzero((left > 0) | (right > 0))
    return float(np.count_nonzero((left > 0) & (right > 0)) / union) if union else 0.0


class UnifiedTrackMatcher:
    """Stable V1/SAM3 identity bridge using mask, box and centre evidence."""

    def __init__(self) -> None:
        self._tracks: dict[str, UnifiedTrack] = {}
        self._next = 1

    def match(self, v1: list[Any], sam: list[dict[str, Any]], frame_index: int) -> list[UnifiedTrack]:
        claimed: set[str] = set()
        for item in v1:
            best: tuple[float, dict[str, Any] | None] = (0.0, None)
            for candidate in sam:
                score = 0.65 * _mask_iou(item.mask, candidate.get("mask")) + 0.35 * _bbox_iou(item.bbox_xyxy, candidate.get("bounding_box"))
                if score > best[0]:
                    best = score, candidate
            candidate_id = best[1].get("sam3_track_id") if best[1] is not None and best[0] >= 0.15 else None
            owner = next((track for track in self._tracks.values() if track.sam3_track_id == candidate_id), None) if candidate_id is not None else None
            existing = owner or next((track for track in self._tracks.values() if track.v1_track_id == item.track_id), None)
            if existing is None and best[1] is not None and best[0] >= 0.15:
                existing = next((track for track in self._tracks.values() if track.sam3_track_id == best[1].get("sam3_track_id")), None)
            if existing is None:
                existing = UnifiedTrack(f"u{self._next}", item.class_name, item.track_id, first_frame=frame_index)
                self._tracks[existing.unified_track_id] = existing
                self._next += 1
            existing.v1_track_id, existing.last_frame = item.track_id, frame_index
            if best[1] is not None and best[0] >= 0.15:
                existing.sam3_track_id = best[1].get("sam3_track_id")
            claimed.add(existing.unified_track_id)
        for candidate in sam:
            if any(track.sam3_track_id == candidate.get("sam3_track_id") for track in self._tracks.values()):
                continue
            track = UnifiedTrack(f"u{self._next}", str(candidate.get("category", "object")), sam3_track_id=candidate.get("sam3_track_id"), first_frame=frame_index, last_frame=frame_index)
            self._tracks[track.unified_track_id] = track
            self._next += 1
        return list(self._tracks.values())


class SAM3Bridge:
    """Thin production bridge to sam3tracking's real adapter and recovery code."""

    def __init__(self, project_root: Path, source: Path, width: int, height: int, device: str, checkpoint: Path | None = None, *, chunk_size: int = 300, chunk_overlap: int = 30) -> None:
        self.project_root, self.source, self.width, self.height, self.device = project_root, source, width, height, device
        self.external_root = Path(os.environ.get("SAM3_REPO_PATH", project_root.parent / "sam3tracking")).expanduser()
        self.checkpoint = checkpoint or Path(os.environ.get("SAM3_MODEL_PATH", self.external_root / "models" / "sam3.pt")).expanduser()
        self._adapter: Any | None = None
        self._next_id = 1
        if chunk_size < 1 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("SAM3 chunking requires chunk_size > chunk_overlap >= 0")
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap
        self.chunk_manifest: list[dict[str, Any]] = []

    def _imports(self):
        configured_python = os.environ.get("SAM3_PYTHON_PATH")
        if configured_python:
            expected = Path(configured_python).expanduser().resolve()
            current = Path(sys.executable).resolve()
            if expected != current:
                raise RuntimeError(
                    "SAM3_PYTHON_PATH points to a different environment. Launch v1 with that interpreter "
                    "(for example: & $env:SAM3_PYTHON_PATH -m surgical_pipeline ...); mixed in-process environments are unsupported."
                )
        source_root = self.external_root / "src"
        if not source_root.is_dir():
            raise RuntimeError(f"sam3tracking source was not found: {source_root}")
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        try:
            # Importing these production components ensures this bridge stays tied
            # to the maintained adapter/recovery/propagation implementation.
            adapter = importlib.import_module("sam3tracking.sam3_adapter").SAM3Adapter
            recovery = importlib.import_module("sam3tracking.recovery_manager").RecoveryManager
            tracker = importlib.import_module("sam3tracking.interactive_tracker").InteractiveTracker
            return adapter, recovery, tracker
        except ImportError as error:
            raise RuntimeError("SAM3 is unavailable. Install sam3tracking's official SAM 3 dependencies and assemble models/sam3.pt.") from error

    @staticmethod
    def chunk_ranges(frame_count: int, chunk_size: int, overlap: int) -> list[tuple[int, int, int]]:
        """Return (source_start, source_end, first_output_frame) with no duplicate output."""
        if frame_count < 0 or chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Invalid SAM3 chunk range configuration")
        ranges: list[tuple[int, int, int]] = []
        start = 0
        while start < frame_count:
            end = min(frame_count, start + chunk_size)
            write_start = 0 if not ranges else ranges[-1][1]
            ranges.append((start, end, write_start))
            if end == frame_count:
                break
            start = end - overlap
        return ranges

    @staticmethod
    def _prompt_point(mask: np.ndarray) -> tuple[int, int] | None:
        binary = np.asarray(mask, dtype=np.uint8)
        if not np.any(binary):
            return None
        distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        return int(x), int(y)

    def _create_tracking_clip(self, source_info: Any, target: Path, start: int, end: int, video_module: Any) -> tuple[int, int]:
        """Create a bounded, optionally downscaled SAM3-only clip; output masks remain source-sized."""
        max_width = max(320, int(os.environ.get("SAM3_TRACKING_MAX_WIDTH", "640")))
        if self.width <= max_width:
            video_module.create_clip(self.source, target, source_info, start, end)
            return self.width, self.height
        tracking_width = max_width
        tracking_height = max(2, round(self.height * tracking_width / self.width / 2) * 2)
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), source_info.fps, (tracking_width, tracking_height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create temporary SAM3 clip: {target}")
        try:
            for _, frame in video_module.read_frames(self.source, start, end):
                writer.write(cv2.resize(frame, (tracking_width, tracking_height), interpolation=cv2.INTER_AREA))
        finally:
            writer.release()
        return tracking_width, tracking_height

    def track(self, events: list[SelectionEvent], frame_count: int) -> dict[int, list[dict[str, Any]]]:
        if not events:
            return {}
        Adapter, RecoveryManager, _InteractiveTracker = self._imports()
        if not self.checkpoint.is_file():
            raise RuntimeError(f"SAM3 checkpoint missing: {self.checkpoint}. Run sam3tracking/scripts/assemble_sam3_model.py.")
        config_module = importlib.import_module("sam3tracking.config")
        metrics_module = importlib.import_module("sam3tracking.metrics")
        video_module = importlib.import_module("sam3tracking.video_io")
        active_events = sorted((item for item in events if item.action in {"add", "select", "resume"}), key=lambda item: item.frame_index)
        for event in active_events:
            obj_id = event.sam3_track_id or self._next_id
            self._next_id = max(self._next_id, obj_id + 1)
            event.sam3_track_id = obj_id
            event.unified_track_id = event.unified_track_id or f"u-sam3-{obj_id}"
        event_by_id = {int(item.sam3_track_id): item for item in active_events if item.sam3_track_id is not None}
        recovery = {
            obj_id: RecoveryManager(config_module.TrackingConfig(), config_module.RecoveryConfig(), self.width, self.height)
            for obj_id in event_by_id
        }
        cache: dict[int, list[dict[str, Any]]] = {}
        predictor: Any | None = None
        source_info = video_module.inspect_video(self.source)
        frame_count = min(frame_count, source_info.frame_count)

        def initialize(adapter: Any, frame_index: int, point: tuple[int, int], obj_id: int) -> list[Any]:
            # The official predictor keeps a long-lived BF16 context while it is
            # built, but autocast is thread-local. Explicitly enter it for point
            # prompts too, matching SAM3Adapter.propagate's CUDA behavior.
            try:
                import torch
                context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else None
            except ImportError:
                context = None
            if context is None:
                return adapter.initialize_target(frame_index, None, points=[point], point_labels=[1], obj_id=obj_id)
            with context:
                return adapter.initialize_target(frame_index, None, points=[point], point_labels=[1], obj_id=obj_id)

        def store(global_index: int, detections: list[Any], write_start: int, *, recovered_ids: set[int] | None = None) -> None:
            if global_index < write_start or global_index >= frame_count:
                return
            for detection in detections:
                obj_id = int(detection.track_id)
                event = event_by_id.get(obj_id)
                if event is None or global_index < event.frame_index:
                    continue
                measured = metrics_module.measure(np.asarray(detection.mask), self.width, self.height)
                recovery[obj_id].assess(detection, measured, global_index)
                if measured.area:
                    recovery[obj_id].recovered(measured)
                self._merge(cache, global_index, self._record(
                    detection, obj_id, event.target_category, event.target_name, event.output_color,
                    global_index, obj_id in (recovered_ids or set()),
                ))

        with tempfile.TemporaryDirectory(prefix="v1_sam3_chunks_") as temporary:
            temp_root = Path(temporary)
            for chunk_index, (start, end, write_start) in enumerate(self.chunk_ranges(frame_count, self.chunk_size, self.chunk_overlap)):
                chunk_started = time.perf_counter()
                entry: dict[str, Any] = {
                    "chunk_index": chunk_index, "source_start_frame": start, "source_end_frame_exclusive": end,
                    "output_start_frame": write_start, "overlap_frames": max(0, write_start - start),
                    "status": "running", "active_target_ids": [], "boundary_reseed_ids": [],
                    "recovery_manager": "sam3tracking.recovery_manager.RecoveryManager",
                    "sam3_internal_image_size": int(os.environ.get("SAM3_INTERNAL_IMAGE_SIZE", "1008")),
                }
                self.chunk_manifest.append(entry)
                carried = [item for item in active_events if item.frame_index < start]
                in_chunk = [item for item in active_events if start <= item.frame_index < end]
                if not carried and not in_chunk:
                    entry.update({
                        "status": "success", "skipped_sam3_session": True,
                        "last_output_frame": None, "last_tracks": [],
                        "elapsed_seconds": time.perf_counter() - chunk_started,
                    })
                    continue
                clip = temp_root / f"chunk_{chunk_index:03d}_{start}_{end}.mp4"
                tracking_width, tracking_height = self._create_tracking_clip(source_info, clip, start, end, video_module)
                entry["sam3_inference_resolution"] = {"width": tracking_width, "height": tracking_height}
                entry["output_mask_resolution"] = {"width": self.width, "height": self.height}
                adapter = Adapter(clip, self.width, self.height, checkpoint=self.checkpoint, device=self.device, predictor=predictor, offload_video_to_cpu=True)
                try:
                    adapter.load()
                    predictor = adapter.predictor
                    reseeded: set[int] = set()
                    for event in carried:
                        obj_id = int(event.sam3_track_id)
                        previous = next((item for item in cache.get(start, []) if int(item.get("sam3_track_id", -1)) == obj_id), None)
                        if previous is None:
                            candidates = [index for index in cache if index <= start and any(int(value.get("sam3_track_id", -1)) == obj_id for value in cache[index])]
                            if candidates:
                                previous = next(value for value in cache[max(candidates)] if int(value.get("sam3_track_id", -1)) == obj_id)
                        point = self._prompt_point(previous["mask"]) if previous is not None else None
                        if point is None:
                            continue
                        initial = initialize(adapter, 0, point, obj_id)
                        reseeded.add(obj_id)
                        recovery[obj_id].record_attempt(start)
                        store(start, initial, write_start, recovered_ids=reseeded)
                    entry["boundary_reseed_ids"] = sorted(reseeded)
                    cursor = 0
                    for boundary in sorted({item.frame_index - start for item in in_chunk}):
                        if boundary > cursor and (reseeded or any(item.frame_index - start <= cursor for item in in_chunk)):
                            for local_index, detections in adapter.propagate(cursor, boundary - cursor):
                                store(start + local_index, detections, write_start)
                        for event in (item for item in in_chunk if item.frame_index - start == boundary):
                            obj_id = int(event.sam3_track_id)
                            initial = initialize(adapter, boundary, event.pixel(self.width, self.height), obj_id)
                            store(event.frame_index, initial, write_start)
                        cursor = boundary
                    has_active = bool(carried or in_chunk)
                    if has_active and cursor < end - start:
                        for local_index, detections in adapter.propagate(cursor, end - start - 1 - cursor):
                            store(start + local_index, detections, write_start)
                    entry["active_target_ids"] = sorted(int(item.sam3_track_id) for item in active_events if item.frame_index < end)
                    available = [index for index in cache if write_start <= index < end]
                    last_frame = max(available) if available else None
                    entry["last_output_frame"] = last_frame
                    entry["last_tracks"] = [
                        {
                            "sam3_track_id": item.get("sam3_track_id"), "target_name": item.get("target_name"),
                            "mask_area_pixels": item.get("mask_area_pixels"), "bounding_box": item.get("bounding_box"),
                            "center": item.get("center"), "visible": item.get("visible"),
                        }
                        for item in (cache.get(last_frame, []) if last_frame is not None else [])
                    ]
                    entry["status"] = "success"
                    entry["gpu_memory_mib"] = adapter.gpu_memory()
                    try:
                        import torch
                        entry["peak_gpu_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else None
                        entry["peak_gpu_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 1024**2) if torch.cuda.is_available() else None
                    except ImportError:
                        entry["peak_gpu_allocated_mib"] = entry["peak_gpu_reserved_mib"] = None
                except Exception as error:
                    entry["status"], entry["error"] = "failed", f"{type(error).__name__}: {error}"
                    raise
                finally:
                    adapter.close()
                    entry["elapsed_seconds"] = time.perf_counter() - chunk_started
        # SAM3 and the per-frame YOLO/pose pass intentionally do not coexist in
        # VRAM on smaller GPUs. The predictor is no longer needed after caching.
        del adapter
        del predictor
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return cache

    @staticmethod
    def _merge(cache: dict[int, list[dict[str, Any]]], frame_index: int, item: dict[str, Any]) -> None:
        values = cache.setdefault(frame_index, [])
        track_id = item["sam3_track_id"]
        values[:] = [value for value in values if value.get("sam3_track_id") != track_id]
        values.append(item)

    @staticmethod
    def _record(item: Any, track_id: int, category: str, target_name: str | None, output_color: str | None, index: int, recovered: bool) -> dict[str, Any]:
        mask = np.asarray(item.mask, dtype=np.uint8)
        center = tuple(map(float, np.median(np.argwhere(mask > 0)[:, ::-1], axis=0))) if np.any(mask) else None
        return {"sam3_track_id": int(getattr(item, "track_id", track_id)), "unified_track_id": f"u-sam3-{track_id}", "category": category, "target_name": target_name or category, "output_color": output_color, "mask": mask, "mask_area_pixels": int(np.count_nonzero(mask)), "bounding_box": getattr(item, "bbox", None), "center": center, "confidence": getattr(item, "confidence", None), "visible": bool(np.any(mask)), "recovered": recovered, "first_frame": index, "last_frame": index}


class InspectionRenderer:
    def __init__(self) -> None:
        self._trails: dict[int, list[tuple[int, int]]] = {}

    @staticmethod
    def _colour(item: dict[str, Any]) -> tuple[int, int, int]:
        colours = {"selected_instrument": (0, 220, 255), "selected_person_1": (255, 220, 0), "selected_person_2": (255, 0, 255)}
        return colours.get(str(item.get("target_name")), (0, 220, 255))

    def render(self, original_frame: np.ndarray, selected: list[dict[str, Any]], tracks: list[UnifiedTrack], mode: str = "overlay") -> np.ndarray:
        if mode == "black-background":
            canvas = np.zeros_like(original_frame)
        elif mode == "blur-background":
            canvas = cv2.GaussianBlur(original_frame, (0, 0), 15)
        elif mode in {"isolate", "mask", "transparent"}:
            canvas = np.zeros_like(original_frame)
        else:
            canvas = original_frame.copy()
        for item in selected:
            colour = self._colour(item)
            track_id = int(item.get("sam3_track_id", 0))
            mask = item.get("mask")
            if mask is not None and np.any(mask):
                if mode in {"isolate", "mask", "transparent"}:
                    canvas[mask > 0] = original_frame[mask > 0] if mode != "mask" else (0, 220, 255)
                else:
                    tint = np.zeros_like(canvas); tint[:] = colour; canvas = np.where((mask > 0)[..., None], cv2.addWeighted(canvas, .36, tint, .64, 0), canvas)
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(canvas, contours, -1, colour, 3, cv2.LINE_AA)
            box = item.get("bounding_box")
            if box:
                x1, y1, x2, y2 = (int(value) for value in box)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 3)
                unified_id = str(item.get("unified_track_id") or next((track.unified_track_id for track in tracks if track.sam3_track_id == track_id), f"SAM3-{track_id}"))
                label = f"{str(item.get('target_name', item.get('category', 'TARGET'))).upper()} | unified_id={unified_id}"
                cv2.putText(canvas, label, (x1, max(28, y1 - 9)), cv2.FONT_HERSHEY_SIMPLEX, .68, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(canvas, label, (x1, max(28, y1 - 9)), cv2.FONT_HERSHEY_SIMPLEX, .68, colour, 2, cv2.LINE_AA)
            center = item.get("center")
            if center is not None:
                trail = self._trails.setdefault(track_id, [])
                trail.append((int(center[0]), int(center[1]))); del trail[:-35]
                for left, right in zip(trail, trail[1:], strict=False):
                    cv2.line(canvas, left, right, colour, 3, cv2.LINE_AA)
        return canvas


class PrivacyXRayRenderer:
    """Synthetic-only renderer. It intentionally has no original-frame parameter."""

    def __init__(self, project_root: Path) -> None:
        configured_root = os.environ.get("XRAY_REPO_PATH")
        self.root = Path(configured_root).expanduser() if configured_root else project_root / "optional_xray"
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        self._ready = False
        self.backend = "builtin"
        self._trails: dict[int, list[tuple[int, int]]] = {}

    @staticmethod
    def _colour(item: dict[str, Any]) -> tuple[int, int, int]:
        return {"selected_instrument": (0, 220, 255), "selected_person_1": (255, 220, 0), "selected_person_2": (255, 0, 255)}.get(str(item.get("target_name")), (0, 220, 255))

    def _load(self) -> None:
        if self._ready:
            return
        if self.root.is_dir():
            try:
                from xray_pipeline.bone_atlas import BoneAtlas
                from xray_pipeline.temporal_pose import TemporalPoseFilter
                from xray_pipeline.skeletal_rig import make_rig
                from xray_pipeline.xray_renderer import render_xray
                self.BoneAtlas, self.TemporalPoseFilter, self.make_rig, self.render_xray = BoneAtlas, TemporalPoseFilter, make_rig, render_xray
                self.atlas = BoneAtlas(self.root / "assets" / "anatomy").load()
                self.temporal = TemporalPoseFilter()
                self.backend = "optional_xray"
            except (ImportError, OSError, ValueError):
                # A clean clone keeps a dependency-free synthetic fallback; an
                # optional vendor renderer can be selected with XRAY_REPO_PATH.
                self.backend = "builtin"
        self._ready = True

    def render(self, size: tuple[int, int], poses: list[Any], instruments: list[Any], selected_ids: set[str], tracks: list[UnifiedTrack], frame_index: int, selected_sam: list[dict[str, Any]]) -> np.ndarray:
        self._load()
        if self.backend == "optional_xray":
            rigs = []
            for pose in poses:
                if pose.pose_track_id is None:
                    continue
                proxy = type("PoseProxy", (), {})()
                proxy.keypoints, proxy.track_id = pose.keypoints.copy(), pose.pose_track_id
                proxy.visible = lambda threshold, points=proxy.keypoints: np.flatnonzero(np.isfinite(points).all(axis=1) & (points[:, 2] >= threshold))
                proxy = self.temporal.update(proxy, frame_index)
                layers, _audit = self.make_rig(proxy, self.atlas, self.temporal, .3, pose.bbox_xyxy)
                rigs.append(layers)
            canvas = self.render_xray(size, rigs, {"background_color": [2, 5, 9], "bone_color_bgr": [244, 247, 234], "glow_color_bgr": [255, 210, 96], "glow_sigma": 7., "glow_alpha": .34, "sharp_alpha": .92})
        else:
            canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            for pose in poses:
                points = np.asarray(pose.keypoints)
                for first, second in COCO17_SKELETON:
                    if first < len(points) and second < len(points) and np.isfinite(points[[first, second], :2]).all() and np.all(points[[first, second], 2] >= .25):
                        cv2.line(canvas, tuple(map(int, points[first, :2])), tuple(map(int, points[second, :2])), (244, 247, 234), 2, cv2.LINE_AA)
                for point in points:
                    if np.isfinite(point[:2]).all() and point[2] >= .25:
                        cv2.circle(canvas, tuple(map(int, point[:2])), 3, (255, 210, 96), -1, cv2.LINE_AA)
        selected_by_sam = {int(item.get("sam3_track_id", -1)): item for item in selected_sam}
        for track in tracks:
            selected = selected_by_sam.get(int(track.sam3_track_id)) if track.sam3_track_id is not None else None
            if selected is None or track.v1_track_id is None:
                continue
            pose = next((item for item in poses if getattr(item, "v1_track_id", None) == track.v1_track_id), None)
            if pose is None:
                continue
            colour = self._colour(selected)
            points = np.asarray(pose.keypoints)
            for first, second in COCO17_SKELETON:
                if first < len(points) and second < len(points) and np.isfinite(points[[first, second], :2]).all() and np.all(points[[first, second], 2] >= .25):
                    cv2.line(canvas, tuple(map(int, points[first, :2])), tuple(map(int, points[second, :2])), colour, 3, cv2.LINE_AA)
            for point in points:
                if np.isfinite(point[:2]).all() and point[2] >= .25:
                    cv2.circle(canvas, tuple(map(int, point[:2])), 4, colour, -1, cv2.LINE_AA)
        for item in selected_sam:
            track_id = int(item.get("sam3_track_id", 0)); colour = self._colour(item); center = item.get("center")
            mask = item.get("mask")
            if mask is not None and np.any(mask):
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(canvas, contours, -1, colour, 2, cv2.LINE_AA)
            if center is not None:
                trail = self._trails.setdefault(track_id, []); trail.append((int(center[0]), int(center[1]))); del trail[:-35]
                for left, right in zip(trail, trail[1:], strict=False): cv2.line(canvas, left, right, colour, 2, cv2.LINE_AA)
                cv2.circle(canvas, trail[-1], 6, colour, -1, cv2.LINE_AA)
            unified_id = str(item.get("unified_track_id") or next((track.unified_track_id for track in tracks if track.sam3_track_id == track_id), f"SAM3-{track_id}"))
            cv2.putText(canvas, f"{str(item.get('target_name', item.get('category', 'TARGET'))).upper()} | {unified_id}", (14, 30 + 28 * max(track_id - 1, 0)), cv2.FONT_HERSHEY_SIMPLEX, .65, colour, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"PERSONNEL: {len(poses)}", (14, size[1] - 16), cv2.FONT_HERSHEY_SIMPLEX, .7, (230, 230, 230), 2, cv2.LINE_AA)
        return canvas


class V1SelectionRecovery:
    """Reject obvious SAM drift and recover the selected instrument from real V1 masks."""

    def __init__(self, events: list[SelectionEvent], width: int, height: int) -> None:
        self.events = {int(event.sam3_track_id): event for event in events if event.sam3_track_id is not None}
        self.last_centers = {track_id: event.pixel(width, height) for track_id, event in self.events.items()}

    def apply(self, raw_tracks: list[dict[str, Any]], instruments: list[Any], frame_index: int) -> list[dict[str, Any]]:
        tracks = [item.copy() for item in raw_tracks]
        for track_id, event in self.events.items():
            if event.target_category != "instrument" or frame_index < event.frame_index:
                continue
            candidates = [
                item for item in instruments
                if event.initial_class_name is None or item.class_name == event.initial_class_name
            ]
            candidates = [item for item in candidates if item.mask is not None and item.centroid() is not None]
            if not candidates:
                continue
            previous_center = self.last_centers[track_id]
            candidate = min(candidates, key=lambda item: float(np.hypot(item.centroid()[0] - previous_center[0], item.centroid()[1] - previous_center[1])))
            candidate_center = candidate.centroid()
            if candidate_center is None:
                continue
            existing = next((item for item in tracks if int(item.get("sam3_track_id", -1)) == track_id), None)
            candidate_area = int(np.count_nonzero(candidate.mask))
            candidate_box = tuple(map(float, candidate.bbox_xyxy)) if candidate.bbox_xyxy else None
            radius = 70.0
            if candidate_box is not None:
                radius = max(radius, .75 * float(np.hypot(candidate_box[2] - candidate_box[0], candidate_box[3] - candidate_box[1])))
            existing_center = existing.get("center") if existing is not None else None
            existing_area = int(existing.get("mask_area_pixels", 0)) if existing is not None else 0
            drifted = (
                existing is None or not existing.get("visible") or existing_center is None
                or float(np.hypot(existing_center[0] - candidate_center[0], existing_center[1] - candidate_center[1])) > radius
                or existing_area > max(5000, candidate_area * 3)
            )
            if not drifted:
                self.last_centers[track_id] = (int(existing_center[0]), int(existing_center[1]))
                continue
            mask = np.asarray(candidate.mask, dtype=np.uint8)
            recovered = {
                "sam3_track_id": track_id, "unified_track_id": event.unified_track_id or f"u-sam3-{track_id}",
                "category": event.target_category, "target_name": event.target_name or event.target_category,
                "output_color": event.output_color, "mask": mask, "mask_area_pixels": candidate_area,
                "bounding_box": candidate.bbox_xyxy, "center": tuple(map(float, candidate_center)),
                "confidence": candidate.confidence, "visible": True, "recovered": True,
                "recovery_method": "v1_instance_mask", "first_frame": frame_index, "last_frame": frame_index,
            }
            if existing is None:
                tracks.append(recovered)
            else:
                tracks[tracks.index(existing)] = recovered
            self.last_centers[track_id] = (int(candidate_center[0]), int(candidate_center[1]))
        return tracks


@dataclass(frozen=True)
class UnifiedRunResult:
    run_dir: Path
    artifacts: list[Path]
    manifest: Path


class UnifiedAnalysisPipeline:
    def __init__(self, root: Path, config: AppConfig) -> None:
        self.root, self.config = root, config

    def run(self, source: Path, output_dir: Path, health_model: Path, instrument_model: Path, pose_model: Path, *, render_mode: RenderMode, enable_sam3: bool = False, enable_xray_skeleton: bool = False, selection_events: list[SelectionEvent] | None = None, progress: Callable[[int, int, float, str], None] | None = None, cancel_event: threading.Event | None = None) -> UnifiedRunResult:
        if render_mode == "legacy":
            raise ValueError("legacy render mode must use the existing pipeline")
        selection_events = selection_events or []
        started = time.perf_counter(); output_dir.mkdir(parents=True, exist_ok=True)
        reader = VideoReader(source); metadata = reader.metadata
        if self.config.max_video_seconds is not None and self.config.max_video_seconds <= 0:
            raise ValueError("max_video_seconds pozitif olmalıdır")
        processing_frame_count = metadata.frame_count
        if self.config.max_video_seconds is not None:
            processing_frame_count = min(metadata.frame_count, max(1, int(np.ceil(self.config.max_video_seconds * metadata.fps))))
        health_path, instrument_path = resolve_model_file_paths(self.root, health_model, instrument_model, auto_download=False)
        device, half = _device(self.config.device, self.config.fp16)
        sam_cache: dict[int, list[dict[str, Any]]] = {}
        sam_chunk_manifest: list[dict[str, Any]] = []
        warnings: list[str] = []
        if enable_sam3 and selection_events:
            bridge = SAM3Bridge(self.root, source, metadata.width, metadata.height, str(device))
            sam_cache = bridge.track(selection_events, processing_frame_count)
            sam_chunk_manifest = bridge.chunk_manifest
        elif enable_sam3:
            warnings.append("SAM3 enabled but no selection events were supplied; clean preview remains unannotated.")
        # Load the per-frame models only after SAM3 has released its predictor;
        # this keeps the real models usable on 6 GiB GPUs without changing output resolution.
        health_model_obj, instrument_model_obj, health_info, instrument_info, health_class = load_validated_yolo_models(health_path, instrument_path, self.config.health_class_name, self.config.health_class_id)
        pose = PoseEstimator(pose_model, device, self.config.fp16, .25); pose_info = pose.load()
        health_tracker = write_tracker_config(output_dir / "health_tracker.yaml", self.config, self.config.health_confidence, self.config.health_iou)
        instrument_tracker = write_tracker_config(output_dir / "instrument_tracker.yaml", self.config, self.config.instrument_confidence, self.config.instrument_iou)
        health_fallback, instrument_fallback = MaskTrackFallback(self.config.track_buffer), MaskTrackFallback(self.config.track_buffer)
        pose_tracker = PoseTracker(max_lost_frames=max(1, int(metadata.fps * self.config.max_lost_seconds)))
        matcher, inspection = UnifiedTrackMatcher(), InspectionRenderer()
        v1_recovery = V1SelectionRecovery(selection_events, metadata.width, metadata.height)
        xray = PrivacyXRayRenderer(self.root) if (render_mode in {"privacy-xray", "dual"} or enable_xray_skeleton) else None
        inspection_writer = VideoWriter(output_dir / "sam3_inspection_silent.mp4", metadata, self.config.output_codec) if render_mode in {"inspection", "dual"} else None
        privacy_writer = VideoWriter(output_dir / "processed_video_xray_sam3_silent.mp4", metadata, self.config.output_codec) if render_mode in {"privacy-xray", "dual"} else None
        health_rows: list[dict[str, Any]] = []; instrument_rows: list[dict[str, Any]] = []; sam_rows: list[dict[str, Any]] = []; unified_rows: list[dict[str, Any]] = []; processed_frames = 0
        try:
            for frame_index, timestamp, frame in reader:
                if frame_index >= processing_frame_count or (self.config.max_video_seconds is not None and timestamp >= self.config.max_video_seconds):
                    break
                if cancel_event and cancel_event.is_set():
                    raise AnalysisCancelled("Unified analysis cancelled")
                health_result = tracked_result(health_model_obj, frame, health_tracker, self.config.health_confidence, self.config.health_iou, device, half, self.config.image_size)
                instrument_result = tracked_result(instrument_model_obj, frame, instrument_tracker, self.config.instrument_confidence, self.config.instrument_iou, device, half, self.config.image_size)
                persons = health_fallback.assign(filter_class(extract_detections(health_result, health_info.names, frame.shape[:2], frame_index, timestamp), health_class), frame_index)
                instruments = instrument_fallback.assign(extract_detections(instrument_result, instrument_info.names, frame.shape[:2], frame_index, timestamp), frame_index)
                poses = pose_tracker.update(match_health_personnel(pose.estimate(frame, frame_index, timestamp), persons), frame_index, .25)
                for pose_item in poses:
                    matching_person = max(persons, key=lambda person: _bbox_iou(pose_item.bbox_xyxy, person.bbox_xyxy), default=None)
                    pose_item.v1_track_id = matching_person.track_id if matching_person and _bbox_iou(pose_item.bbox_xyxy, matching_person.bbox_xyxy) >= .05 else None
                sam_tracks = v1_recovery.apply(sam_cache.get(frame_index, []), instruments, frame_index)
                tracks = matcher.match(persons + instruments, sam_tracks, frame_index)
                active = [item for item in selection_events if item.frame_index <= frame_index and item.action not in {"remove", "pause"}]
                selected = {item.unified_track_id for item in active if item.unified_track_id}
                for track in tracks:
                    track.selected = track.unified_track_id in selected or any(item.sam3_track_id == track.sam3_track_id for item in active)
                selected_sam = [item for item in sam_tracks if any(event.sam3_track_id == item.get("sam3_track_id") for event in active)]
                analysis = FrameAnalysis(frame_index, timestamp, metadata.width, metadata.height, persons, instruments, poses, [item.track_id for item in persons + instruments if item.track_id is not None], sam_tracks, tracks, [item.unified_track_id for item in tracks if item.selected], active)
                if inspection_writer:
                    inspection_writer.write(inspection.render(frame, selected_sam, tracks))
                if privacy_writer:
                    # The source RGB frame is intentionally NOT passed to this renderer.
                    privacy_writer.write(xray.render((metadata.width, metadata.height), poses, instruments, set(analysis.selected_tracks), tracks, frame_index, selected_sam))  # type: ignore[union-attr]
                health_rows.extend(_detection_rows(persons, "person")); instrument_rows.extend(_detection_rows(instruments, "instrument"))
                sam_rows.extend(_sam_rows(sam_tracks, frame_index, timestamp)); unified_rows.extend(_unified_rows(tracks, frame_index, timestamp))
                processed_frames += 1
                if progress and (frame_index == 0 or frame_index % 5 == 0 or frame_index + 1 == processing_frame_count):
                    elapsed = time.perf_counter() - started; progress(frame_index + 1, processing_frame_count, elapsed / max(frame_index + 1, 1) * max(processing_frame_count - frame_index - 1, 0), "Unified analysis")
        except Exception:
            if inspection_writer: inspection_writer.abort()
            if privacy_writer: privacy_writer.abort()
            raise
        finally:
            reader.close()
        artifacts: list[Path] = []
        if inspection_writer:
            inspection_writer.close(); inspection_writer.finalise(source, output_dir / "sam3_inspection.mp4", keep_audio=True); artifacts.append(output_dir / "sam3_inspection.mp4")
        if privacy_writer:
            privacy_writer.close(); privacy_writer.finalise(source, output_dir / "processed_video_xray_sam3.mp4", keep_audio=False); artifacts.append(output_dir / "processed_video_xray_sam3.mp4")
        # Preserve legacy consumer expectation in dual mode without duplicating inference.
        if privacy_writer:
            target = output_dir / "processed_video.mp4"
            if not target.exists():
                shutil.copyfile(output_dir / "processed_video_xray_sam3.mp4", target)
            artifacts.append(target)
            # Stable, explicit name for the final shareable privacy artifact.
            security_alias = output_dir / "security_xray_final.mp4"
            if not security_alias.exists():
                shutil.copyfile(output_dir / "processed_video_xray_sam3.mp4", security_alias)
            artifacts.append(security_alias)
        if inspection_writer and privacy_writer:
            comparison = _write_comparison_video(output_dir / "sam3_inspection.mp4", output_dir / "processed_video_xray_sam3.mp4", output_dir / "tracking_xray_comparison.mp4")
            artifacts.append(comparison)
            inspection_alias = output_dir / "inspection_tracking.mp4"; shutil.copyfile(output_dir / "sam3_inspection.mp4", inspection_alias); artifacts.append(inspection_alias)
            privacy_alias = output_dir / "privacy_xray_tracking.mp4"; shutil.copyfile(output_dir / "processed_video_xray_sam3.mp4", privacy_alias); artifacts.append(privacy_alias)
        paths = {"health_person_tracks.csv": health_rows, "instrument_tracks.csv": instrument_rows, "sam3_tracks.csv": sam_rows, "unified_tracks.csv": unified_rows}
        for name, rows in paths.items():
            _write_csv(output_dir / name, rows); artifacts.append(output_dir / name)
        events_path = output_dir / "selection_events.json"; write_json(events_path, [asdict(item) for item in selection_events]); artifacts.append(events_path)
        unified_by_sam = {event.sam3_track_id: event.unified_track_id for event in selection_events if event.sam3_track_id is not None}
        for chunk in sam_chunk_manifest:
            chunk["unified_track_ids"] = {str(track_id): unified_by_sam.get(track_id) for track_id in chunk.get("active_target_ids", [])}
        chunk_manifest_path = output_dir / "chunk_manifest.json"; write_json(chunk_manifest_path, {"chunk_size_frames": 300, "overlap_frames": 30, "duplicate_output_frames": 0, "chunks": sam_chunk_manifest}); artifacts.append(chunk_manifest_path)
        write_json(output_dir / "events.json", {"selection_events": len(selection_events), "warnings": warnings}); artifacts.append(output_dir / "events.json")
        summary = {"schema_version": "2.0", "render_mode": render_mode, "processed_frames": processed_frames, "total_people": len({row.get("v1_track_id") for row in health_rows if row.get("v1_track_id") is not None}), "warnings": warnings}; write_json(output_dir / "summary.json", summary); artifacts.append(output_dir / "summary.json")
        quality = {"schema_version": "2.0", "pose_model": pose_info.safe_dict(), "sam3_enabled": enable_sam3, "xray_skeleton_enabled": bool(xray), "warnings": warnings}; write_json(output_dir / "quality_report.json", quality); artifacts.append(output_dir / "quality_report.json")
        privacy = {"schema_version": "2.0", "original_rgb_used_by_privacy_renderer": False, "privacy_renderer_used_original_rgb": False, "source_rgb_copied_to_privacy_output": False, "audio_copied": False, "renderer_contract": "PrivacyXRayRenderer.render accepts no original frame", "passed": True}; write_json(output_dir / "privacy_report.json", privacy); artifacts.append(output_dir / "privacy_report.json")
        config_path = output_dir / "run_config.yaml"; config_path.write_text("render_mode: " + render_mode + "\nsam3_enabled: " + str(enable_sam3).lower() + "\nxray_skeleton_enabled: " + str(bool(xray)).lower() + "\n", encoding="utf-8"); artifacts.append(config_path)
        manifest = {"schema_version": "2.0", "status": "success", "input_video": {"filename": source.name, "sha256": sha256_file(source), **metadata.safe_dict()}, "processed_frame_count": processed_frames, "models": {"health": health_info.safe_dict(), "instrument": instrument_info.safe_dict(), "pose": pose_info.safe_dict()}, "sam3_enabled": enable_sam3, "xray_skeleton_enabled": bool(xray), "render_mode": render_mode, "privacy_renderer_used_original_rgb": False, "source_rgb_copied_to_privacy_output": False, "audio_copied": False, "selection_event_count": len(selection_events), "output_files": {path.name: sha256_file(path) for path in artifacts if path.is_file()}, "warnings": warnings, "elapsed_seconds": time.perf_counter() - started}
        manifest_path = output_dir / "run_manifest.json"; write_json(manifest_path, manifest); artifacts.append(manifest_path)
        return UnifiedRunResult(output_dir, artifacts, manifest_path)


def _detection_rows(items: list[Any], kind: str) -> list[dict[str, Any]]:
    return [{"frame_index": item.frame_index, "timestamp": item.timestamp_s, "kind": kind, "v1_track_id": item.track_id, "class_name": item.class_name, "confidence": item.confidence, "bounding_box": list(item.bbox_xyxy) if item.bbox_xyxy else None, "center": item.centroid()} for item in items]

def _sam_rows(items: list[dict[str, Any]], index: int, timestamp: float) -> list[dict[str, Any]]:
    return [{key: value for key, value in item.items() if key != "mask"} | {"frame_index": index, "timestamp": timestamp} for item in items]

def _unified_rows(items: list[UnifiedTrack], index: int, timestamp: float) -> list[dict[str, Any]]:
    return [asdict(item) | {"frame_index": index, "timestamp": timestamp} for item in items]

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["frame_index"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, tuple, dict)) else value for key, value in row.items()})


def _write_comparison_video(inspection_path: Path, privacy_path: Path, destination: Path) -> Path:
    """Write a synchronized, labeled comparison without exposing RGB on the privacy panel."""
    left, right = cv2.VideoCapture(str(inspection_path)), cv2.VideoCapture(str(privacy_path))
    try:
        width, height, fps = int(left.get(cv2.CAP_PROP_FRAME_WIDTH)), int(left.get(cv2.CAP_PROP_FRAME_HEIGHT)), left.get(cv2.CAP_PROP_FPS)
        if width <= 0 or height <= 0 or fps <= 0:
            raise RuntimeError("Comparison inputs could not be opened.")
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height))
        if not writer.isOpened():
            raise RuntimeError("Comparison video writer could not be opened.")
        try:
            frame_index = 0
            while True:
                ok_left, frame_left = left.read(); ok_right, frame_right = right.read()
                if not ok_left or not ok_right:
                    break
                cv2.rectangle(frame_left, (0, 0), (width, 38), (0, 0, 0), -1)
                cv2.rectangle(frame_right, (0, 0), (width, 38), (0, 0, 0), -1)
                cv2.putText(frame_left, "SAM3 INSPECTION TRACKING", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 220, 255), 2, cv2.LINE_AA)
                cv2.putText(frame_right, "PRIVACY X-RAY SKELETON", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 220, 0), 2, cv2.LINE_AA)
                stamp = f"frame={frame_index}  t={frame_index / fps:06.2f}s"
                cv2.putText(frame_left, stamp, (width - 245, 26), cv2.FONT_HERSHEY_SIMPLEX, .55, (235, 235, 235), 1, cv2.LINE_AA)
                cv2.putText(frame_right, stamp, (width - 245, 26), cv2.FONT_HERSHEY_SIMPLEX, .55, (235, 235, 235), 1, cv2.LINE_AA)
                writer.write(np.hstack((frame_left, frame_right)))
                frame_index += 1
        finally:
            writer.release()
    finally:
        left.release(); right.release()
    return destination
