"""Optional local monocular depth estimation using Depth Anything V2 Small."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


class DepthModelError(RuntimeError):
    """Depth is enabled but its explicitly requested model cannot be used."""


@dataclass
class DepthStatus:
    enabled: bool
    ready: bool
    detail: str


class DepthEstimator:
    """Depth Anything V2 Small wrapper.

    The model produces relative inverse-depth-like values, never metres or centimetres.
    Hugging Face caches downloaded files locally; normal analysis afterwards is local.
    """

    def __init__(self, model_id: str, device: str = "auto", enabled: bool = True) -> None:
        self.model_id = model_id
        self.enabled = enabled
        self.device = self._resolve_device(device)
        self.processor = None
        self.model = None
        self._last_depth: np.ndarray | None = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        try:
            import torch

            if device == "auto":
                return "cuda:0" if torch.cuda.is_available() else "cpu"
            return device
        except ImportError:
            return "cpu"

    def status(self) -> DepthStatus:
        if not self.enabled:
            return DepthStatus(False, True, "Kapalı (kullanıcı tercihi)")
        if self.model is not None:
            return DepthStatus(True, True, f"Hazır: {self.model_id} ({self.device})")
        return DepthStatus(True, False, f"Yüklenecek: {self.model_id}")

    def load(self) -> None:
        if not self.enabled or self.model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id).to(self.device).eval()
            self._torch = torch
        except Exception as error:
            raise DepthModelError(
                f"Derinlik modeli ({self.model_id}) yüklenemedi. İnternet/cache ve transformers kurulumunu kontrol edin: {error}"
            ) from error

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self.enabled:
            raise DepthModelError("Derinlik hesaplama kapalı.")
        self.load()
        assert self.processor is not None and self.model is not None
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            output = self.model(**inputs)
            predicted = output.predicted_depth.unsqueeze(1)
            predicted = self._torch.nn.functional.interpolate(predicted, size=frame_bgr.shape[:2], mode="bicubic", align_corners=False)
        depth = predicted.squeeze().float().cpu().numpy().astype(np.float32)
        # Robust per-frame normalisation, documented as relative depth only.
        finite = depth[np.isfinite(depth)]
        if finite.size:
            low, high = np.percentile(finite, [2, 98])
            depth = np.clip((depth - low) / max(high - low, 1e-6), 0.0, 1.0)
        self._last_depth = depth
        return depth

    def cached_or_estimate(self, frame_bgr: np.ndarray, frame_index: int, stride: int) -> tuple[np.ndarray, bool]:
        """Return a depth map; between stride frames, reuse the last map and mark it interpolated."""
        if frame_index % stride == 0 or self._last_depth is None:
            return self.estimate(frame_bgr), True
        return self._last_depth, False
