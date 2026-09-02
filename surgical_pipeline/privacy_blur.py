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
        alpha = cv2.GaussianBlur(binary.astype(np.float32), (blur_size, blur_size), feather_px / 2)
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


class PrivacyMasker:
    """Anonymise only health detections, with bounded tracker-based mask persistence."""

    def __init__(self, mode: str = "gaussian", blur_kernel: int = 51, pixelation_factor: int = 14, dilation_px: int = 13, feather_px: int = 9, persistence_frames: int = 8) -> None:
        self.mode = mode
        self.blur_kernel = blur_kernel if blur_kernel % 2 else blur_kernel + 1
        self.pixelation_factor = pixelation_factor
        self.dilation_px = dilation_px
        self.feather_px = feather_px
        self.persistence_frames = persistence_frames
        self._cache: dict[int, CachedMask] = {}

    def _blurred(self, frame: np.ndarray) -> np.ndarray:
        if self.mode == "pixelate":
            return pixelate(frame, self.pixelation_factor)
        return cv2.GaussianBlur(frame, (self.blur_kernel, self.blur_kernel), 0)

    def apply(self, frame: np.ndarray, detections: list[Detection], frame_index: int) -> np.ndarray:
        """Apply privacy effect and expire stale masks before each frame is released."""
        aggregate = np.zeros(frame.shape[:2], dtype=np.float32)
        current_ids: set[int] = set()
        for detection in detections:
            if detection.track_id is None:
                continue
            current_ids.add(detection.track_id)
            self._cache[detection.track_id] = CachedMask(detection.mask.copy(), frame_index)
            aggregate = np.maximum(aggregate, feather_mask(detection.mask, self.dilation_px, self.feather_px))
        for track_id, cached in list(self._cache.items()):
            age = frame_index - cached.last_seen_frame
            if age > self.persistence_frames:
                del self._cache[track_id]
            elif track_id not in current_ids:
                # Only a few frames of persistence; no unbounded stale background blur.
                fade = max(0.0, 1.0 - age / (self.persistence_frames + 1))
                aggregate = np.maximum(aggregate, feather_mask(cached.mask, self.dilation_px, self.feather_px) * fade)
        if not np.any(aggregate > 0):
            return frame
        blurred = self._blurred(frame)
        alpha = aggregate[..., None]
        return (frame.astype(np.float32) * (1.0 - alpha) + blurred.astype(np.float32) * alpha).astype(np.uint8)
