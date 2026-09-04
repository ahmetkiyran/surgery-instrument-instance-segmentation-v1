"""Segmentation-mask-based personnel anonymisation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .schemas import Detection


def _kernel(size: int) -> np.ndarray:
    size = max(1, int(size))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def feather_mask(mask: np.ndarray, dilation_px: int, feather_px: int) -> np.ndarray:
    """Expand a binary mask then turn its edge into a smooth alpha matte."""
    binary = (mask > 0).astype(np.uint8)
    if dilation_px > 0:
        binary = cv2.dilate(binary, _kernel(2 * dilation_px + 1), iterations=1)
    if feather_px > 0:
        blur_size = max(3, feather_px * 2 + 1)
        # Keep alpha within the (already dilated) privacy region. A Gaussian blur
        # alone leaks partial blur outside a protected mask's intended boundary.
        alpha = cv2.GaussianBlur(binary.astype(np.float32), (blur_size, blur_size), feather_px / 2) * binary
    else:
        alpha = binary.astype(np.float32)
    return np.clip(alpha, 0.0, 1.0)


def pixelate(image: np.ndarray, factor: int) -> np.ndarray:
    height, width = image.shape[:2]
    small = cv2.resize(image, (max(1, width // factor), max(1, height // factor)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


@dataclass
class CachedMask:
    mask: np.ndarray
    last_seen_frame: int


@dataclass(frozen=True)
class PrivacyApplication:
    frame: np.ndarray
    mask_instances: int
    bbox_fallback_instances: int
    persisted_instances: int


class PrivacyMasker:
    """Anonymise only health detections, with bounded tracker-based mask persistence."""

    def __init__(self, mode: str = "gaussian", blur_kernel: int = 51, pixelation_factor: int = 14, dilation_px: int = 13, feather_px: int = 9, persistence_frames: int = 8, bbox_fallback: bool = True) -> None:
        self.mode = mode
        self.blur_kernel = blur_kernel if blur_kernel % 2 else blur_kernel + 1
        self.pixelation_factor = pixelation_factor
        self.dilation_px = dilation_px
        self.feather_px = feather_px
        self.persistence_frames = persistence_frames
        self.bbox_fallback = bbox_fallback
        self._cache: dict[int, CachedMask] = {}

    def _blurred(self, frame: np.ndarray) -> np.ndarray:
        if self.mode == "pixelate":
            return pixelate(frame, self.pixelation_factor)
        return cv2.GaussianBlur(frame, (self.blur_kernel, self.blur_kernel), 0)

    @staticmethod
    def _bbox_mask(shape: tuple[int, int], bbox: tuple[float, float, float, float] | None) -> np.ndarray | None:
        if bbox is None:
            return None
        height, width = shape
        x1, y1, x2, y2 = bbox
        left, top = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        right, bottom = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
        if right <= left or bottom <= top:
            return None
        mask = np.zeros(shape, dtype=np.uint8)
        mask[top:bottom, left:right] = 1
        return mask

    def apply_with_quality(self, frame: np.ndarray, detections: list[Detection], frame_index: int) -> PrivacyApplication:
        """Apply privacy effect and expire stale masks before each frame is released."""
        aggregate = np.zeros(frame.shape[:2], dtype=np.float32)
        current_ids: set[int] = set()
        mask_instances = bbox_instances = persisted_instances = 0
        for detection in detections:
            mask = (detection.mask > 0).astype(np.uint8)
            if np.any(mask):
                mask_instances += 1
            elif self.bbox_fallback:
                mask = self._bbox_mask(frame.shape[:2], detection.bbox_xyxy)
                if mask is None:
                    continue
                bbox_instances += 1
            else:
                continue
            if detection.track_id is not None:
                current_ids.add(detection.track_id)
                self._cache[detection.track_id] = CachedMask(mask.copy(), frame_index)
            aggregate = np.maximum(aggregate, feather_mask(mask, self.dilation_px, self.feather_px))
        for track_id, cached in list(self._cache.items()):
            age = frame_index - cached.last_seen_frame
            if age > self.persistence_frames:
                del self._cache[track_id]
            elif track_id not in current_ids:
                # Only a few frames of persistence; no unbounded stale background blur.
                fade = max(0.0, 1.0 - age / (self.persistence_frames + 1))
                aggregate = np.maximum(aggregate, feather_mask(cached.mask, self.dilation_px, self.feather_px) * fade)
                persisted_instances += 1
        if not np.any(aggregate > 0):
            return PrivacyApplication(frame, mask_instances, bbox_instances, persisted_instances)
        blurred = self._blurred(frame)
        alpha = aggregate[..., None]
        result = (frame.astype(np.float32) * (1.0 - alpha) + blurred.astype(np.float32) * alpha).astype(np.uint8)
        return PrivacyApplication(result, mask_instances, bbox_instances, persisted_instances)

    def apply(self, frame: np.ndarray, detections: list[Detection], frame_index: int) -> np.ndarray:
        """Compatibility wrapper returning only the anonymised frame."""
        return self.apply_with_quality(frame, detections, frame_index).frame
