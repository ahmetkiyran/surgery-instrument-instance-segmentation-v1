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
    if boxes is None or masks is None or getattr(masks, "data", None) is None:
        return []
    classes = _as_numpy(getattr(boxes, "cls", None))
    confidences = _as_numpy(getattr(boxes, "conf", None))
    ids = _as_numpy(getattr(boxes, "id", None))
    raw_masks = _as_numpy(masks.data)
    if classes is None or confidences is None or raw_masks is None:
        return []
    height, width = frame_shape
    detections: list[Detection] = []
    for index, (class_id, confidence, raw_mask) in enumerate(zip(classes, confidences, raw_masks, strict=False)):
        mask = (raw_mask > 0.5).astype(np.uint8)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        track_id = int(ids[index]) if ids is not None and index < len(ids) and np.isfinite(ids[index]) else None
        class_number = int(class_id)
        detections.append(Detection(class_number, names.get(class_number, str(class_number)), float(confidence), track_id, mask, frame_index, timestamp_s))
    return detections


def filter_class(detections: list[Detection], class_id: int) -> list[Detection]:
    return [detection for detection in detections if detection.class_id == class_id]
