"""Configuration loading and validated runtime overrides."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .utils import load_dotenv_file


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


@dataclass
class AppConfig:
    output_dir: Path = Path("outputs")
    device: str = "auto"
    save_audio: bool = True
    max_video_seconds: float | None = None
    privacy_mode: str = "gaussian"
    blur_kernel: int = 51
    pixelation_factor: int = 14
    mask_dilation_px: int = 13
    feather_px: int = 9
    persistence_frames: int = 8
    max_persistence_iou: float = 0.15
    tracker_algorithm: str = "botsort"
    confidence: float = 0.35
    iou: float = 0.45
    health_confidence: float = 0.35
    health_iou: float = 0.45
    instrument_confidence: float = 0.25
    instrument_iou: float = 0.40
    track_buffer: int = 30
    min_confirm_frames: int = 3
    max_lost_seconds: float = 0.75
    max_gap_seconds: float = 0.5
    smoothing_window: int = 3
    max_relative_jump: float = 0.35
    depth_enabled: bool = True
    depth_model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    depth_stride: int = 3
    depth_device: str = "auto"
    camera: dict[str, float | None] = field(default_factory=dict)
    health_model_path: Path | None = None
    instrument_model_path: Path | None = None
    instrument_display_names: dict[str, str] = field(default_factory=dict)

    def validate(self) -> "AppConfig":
        if self.privacy_mode not in {"gaussian", "pixelate"}:
            raise ValueError("privacy.mode gaussian veya pixelate olmalıdır.")
        if self.tracker_algorithm not in {"botsort", "bytetrack"}:
            raise ValueError("tracking.algorithm botsort veya bytetrack olmalıdır.")
        if not 0 < self.confidence <= 1 or not 0 < self.iou <= 1:
            raise ValueError("Confidence ve IoU 0 ile 1 arasında olmalıdır.")
        if self.blur_kernel < 3:
            raise ValueError("Blur çekirdeği en az 3 olmalıdır.")
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1
        if self.depth_stride < 1 or self.min_confirm_frames < 1:
            raise ValueError("Depth stride ve doğrulama karesi en az 1 olmalıdır.")
        if self.max_gap_seconds < 0 or self.max_lost_seconds < 0:
            raise ValueError("Zaman boşlukları negatif olamaz.")
        return self

    def public_run_config(self) -> dict[str, Any]:
        """Return persisted config without machine-specific paths."""
        return {
            "runtime": {"device": self.device, "save_audio": self.save_audio, "max_video_seconds": self.max_video_seconds},
            "privacy": {"mode": self.privacy_mode, "blur_kernel": self.blur_kernel, "pixelation_factor": self.pixelation_factor, "mask_dilation_px": self.mask_dilation_px, "feather_px": self.feather_px, "persistence_frames": self.persistence_frames},
            "tracking": {"algorithm": self.tracker_algorithm, "confidence": self.confidence, "iou": self.iou, "track_buffer": self.track_buffer, "min_confirm_frames": self.min_confirm_frames, "max_lost_seconds": self.max_lost_seconds},
            "analytics": {"max_gap_seconds": self.max_gap_seconds, "smoothing_window": self.smoothing_window, "max_relative_jump": self.max_relative_jump},
            "depth": {"enabled": self.depth_enabled, "model_id": self.depth_model_id, "depth_stride": self.depth_stride, "device": self.depth_device},
            "camera": self.camera,
        }


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Yapılandırma dosyası nesne olmalıdır: {path.name}")
    return loaded


def load_config(project_root: Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    root = (project_root or Path.cwd()).resolve()
    data = _read_yaml(root / "configs" / "default.yaml")
    data = deepcopy(data)
    if overrides:
        _merge(data, overrides)
    env = load_dotenv_file(root / ".env")
    runtime = data.get("runtime", {})
    privacy = data.get("privacy", {})
    tracking = data.get("tracking", {})
    depth = data.get("depth", {})
    analytics = data.get("analytics", {})
    health = tracking.get("health", {})
    instrument = tracking.get("instrument", {})
    names = _read_yaml(root / "configs" / "instrument_names.yaml").get("instrument_display_names", {})
    config = AppConfig(
        output_dir=Path(runtime.get("output_dir", "outputs")),
        device=str(runtime.get("device", "auto")), save_audio=bool(runtime.get("save_audio", True)), max_video_seconds=runtime.get("max_video_seconds"),
        privacy_mode=str(privacy.get("mode", "gaussian")), blur_kernel=int(privacy.get("blur_kernel", 51)), pixelation_factor=int(privacy.get("pixelation_factor", 14)), mask_dilation_px=int(privacy.get("mask_dilation_px", 13)), feather_px=int(privacy.get("feather_px", 9)), persistence_frames=int(privacy.get("persistence_frames", 8)), max_persistence_iou=float(privacy.get("max_persistence_iou", 0.15)),
        tracker_algorithm=str(tracking.get("algorithm", "botsort")), confidence=float(tracking.get("confidence", 0.35)), iou=float(tracking.get("iou", 0.45)), health_confidence=float(health.get("confidence", tracking.get("confidence", 0.35))), health_iou=float(health.get("iou", tracking.get("iou", 0.45))), instrument_confidence=float(instrument.get("confidence", tracking.get("confidence", 0.25))), instrument_iou=float(instrument.get("iou", tracking.get("iou", 0.40))), track_buffer=int(tracking.get("track_buffer", 30)), min_confirm_frames=int(tracking.get("min_confirm_frames", 3)), max_lost_seconds=float(tracking.get("max_lost_seconds", 0.75)),
        max_gap_seconds=float(analytics.get("max_gap_seconds", 0.5)), smoothing_window=int(analytics.get("smoothing_window", 3)), max_relative_jump=float(analytics.get("max_relative_jump", 0.35)),
        depth_enabled=bool(depth.get("enabled", True)), depth_model_id=str(depth.get("model_id", "depth-anything/Depth-Anything-V2-Small-hf")), depth_stride=int(depth.get("depth_stride", 3)), depth_device=str(depth.get("device", "auto")),
        camera=data.get("camera", {}),
        health_model_path=Path(env["HEALTH_MODEL_PATH"]).expanduser() if env.get("HEALTH_MODEL_PATH") else None,
        instrument_model_path=Path(env["INSTRUMENT_MODEL_PATH"]).expanduser() if env.get("INSTRUMENT_MODEL_PATH") else None,
        instrument_display_names={str(k): str(v) for k, v in names.items()},
    )
    return config.validate()


def write_run_config(path: Path, config: AppConfig) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.public_run_config(), handle, allow_unicode=True, sort_keys=False)
