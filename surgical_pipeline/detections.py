"""Conversion from Ultralytics results to package-level detection records."""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np

from .schemas import Detection


def _as_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def extract_detections(result, names: Mapping[int, str], frame_shape: tuple[int, int], frame_index: int, timestamp_s: float) -> list[Detection]:
    """Extract masks, confidence and model-owned track IDs without drawing anything."""
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None:
        return []
    classes = _as_numpy(getattr(boxes, "cls", None))
    confidences = _as_numpy(getattr(boxes, "conf", None))
    ids = _as_numpy(getattr(boxes, "id", None))
    raw_masks = _as_numpy(getattr(masks, "data", None)) if masks is not None else None
    boxes_xyxy = _as_numpy(getattr(boxes, "xyxy", None))
    if classes is None or confidences is None:
        return []
    height, width = frame_shape
    detections: list[Detection] = []
    for index, (class_id, confidence) in enumerate(zip(classes, confidences, strict=False)):
        quality = "valid"
        if raw_masks is not None and index < len(raw_masks):
            mask = (raw_masks[index] > 0.5).astype(np.uint8)
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            if not np.any(mask):
                quality = "empty_mask"
        else:
            mask = np.zeros((height, width), dtype=np.uint8)
            quality = "missing_mask"
        bbox = None
        if boxes_xyxy is not None and index < len(boxes_xyxy) and len(boxes_xyxy[index]) >= 4:
            x1, y1, x2, y2 = (float(value) for value in boxes_xyxy[index][:4])
            bbox = (max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2))
        track_id = int(ids[index]) if ids is not None and index < len(ids) and np.isfinite(ids[index]) else None
        class_number = int(class_id)
        tracking_state = "model_track" if track_id is not None else "unassigned"
        detections.append(Detection(class_number, names.get(class_number, str(class_number)), float(confidence), track_id, mask, frame_index, timestamp_s, bbox, quality, tracking_state))
    return detections


def filter_class(detections: list[Detection], class_id: int) -> list[Detection]:
    return [detection for detection in detections if detection.class_id == class_id]
