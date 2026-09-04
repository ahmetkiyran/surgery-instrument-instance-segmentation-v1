"""Configuration loading and validated runtime overrides."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
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
    bbox_privacy_fallback: bool = True
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
    minimum_interval_seconds: float = 0.0
    smoothing_window: int = 3
    max_relative_jump: float = 0.35
    depth_enabled: bool = True
    depth_model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    depth_stride: int = 3
    depth_device: str = "auto"
    fp16: bool = True
    image_size: int | None = None
    random_seed: int | None = 0
    log_level: str = "INFO"
    output_codec: str = "mp4v"
    allow_cpu_fallback: bool = False
    health_class_name: str | None = None
    health_class_id: int | None = None
    camera: dict[str, float | None] = field(default_factory=dict)
    health_model_path: Path | None = None
    instrument_model_path: Path | None = None
    pose_model_path: Path | None = None
    cli_privacy_mode: str = "skeleton-only"
    instrument_display_names: dict[str, str] = field(default_factory=dict)

    def validate(self) -> "AppConfig":
        if self.privacy_mode not in {"gaussian", "pixelate"}:
            raise ValueError("privacy.mode gaussian veya pixelate olmalıdır.")
        if self.tracker_algorithm not in {"botsort", "bytetrack"}:
            raise ValueError("tracking.algorithm botsort veya bytetrack olmalıdır.")
        if not 0 < self.confidence <= 1 or not 0 < self.iou <= 1:
            raise ValueError("Confidence ve IoU 0 ile 1 arasında olmalıdır.")
        if not all(0 < value <= 1 for value in (self.health_confidence, self.instrument_confidence, self.health_iou, self.instrument_iou)):
            raise ValueError("Health/instrument confidence ve IoU 0 ile 1 arasında olmalıdır.")
        if self.blur_kernel < 3:
            raise ValueError("Blur çekirdeği en az 3 olmalıdır.")
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1
        if self.depth_stride < 1 or self.min_confirm_frames < 1:
            raise ValueError("Depth stride ve doğrulama karesi en az 1 olmalıdır.")
        if self.max_gap_seconds < 0 or self.minimum_interval_seconds < 0 or self.max_lost_seconds < 0:
            raise ValueError("Zaman boşlukları negatif olamaz.")
        if self.image_size is not None and self.image_size < 32:
            raise ValueError("Görüntü boyutu en az 32 olmalıdır.")
        if self.output_codec.casefold() not in {"mp4v", "avc1", "h264"}:
            raise ValueError("output_codec mp4v, avc1 veya h264 olmalıdır.")
        if self.health_class_id is not None and self.health_class_id < 0:
            raise ValueError("health_class_id negatif olamaz.")
        if self.health_class_id is not None and self.health_class_name is not None:
            raise ValueError("Health class için aynı anda hem isim hem ID override verilemez.")
        device = str(self.device).strip().casefold()
        if not (device in {"auto", "cpu"} or device.isdecimal() or re.fullmatch(r"cuda(?::\d+)?", device)):
            raise ValueError("runtime.device auto, cpu, bir GPU indeksi veya cuda[:N] olmalıdır.")
        if self.cli_privacy_mode not in {"skeleton-only", "blur", "both"}:
            raise ValueError("cli.privacy_mode skeleton-only, blur veya both olmalıdır.")
        self.log_level = self.log_level.upper()
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("log_level DEBUG, INFO, WARNING veya ERROR olmalıdır.")
        return self

    def public_run_config(self) -> dict[str, Any]:
        """Return persisted config without machine-specific paths."""
        return {
            "runtime": {"device": self.device, "save_audio": self.save_audio, "max_video_seconds": self.max_video_seconds},
            "privacy": {"mode": self.privacy_mode, "blur_kernel": self.blur_kernel, "pixelation_factor": self.pixelation_factor, "mask_dilation_px": self.mask_dilation_px, "feather_px": self.feather_px, "persistence_frames": self.persistence_frames, "bbox_privacy_fallback": self.bbox_privacy_fallback},
            "tracking": {"algorithm": self.tracker_algorithm, "confidence": self.confidence, "iou": self.iou, "track_buffer": self.track_buffer, "min_confirm_frames": self.min_confirm_frames, "max_lost_seconds": self.max_lost_seconds},
            "analytics": {"max_gap_seconds": self.max_gap_seconds, "minimum_interval_seconds": self.minimum_interval_seconds, "smoothing_window": self.smoothing_window, "max_relative_jump": self.max_relative_jump},
            "depth": {"enabled": self.depth_enabled, "model_id": self.depth_model_id, "depth_stride": self.depth_stride, "device": self.depth_device},
            "model": {"health_class_name": self.health_class_name, "health_class_id": self.health_class_id, "image_size": self.image_size, "fp16": self.fp16, "allow_cpu_fallback": self.allow_cpu_fallback},
            "cli": {"privacy_mode": self.cli_privacy_mode},
            "reproducibility": {"random_seed": self.random_seed, "log_level": self.log_level, "output_codec": self.output_codec},
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


def _source_value(
    default_data: dict[str, Any],
    file_data: dict[str, Any],
    overrides: dict[str, Any],
    section: str,
    key: str,
    environment: dict[str, str],
    environment_key: str,
    fallback: Any,
) -> Any:
    """Resolve explicit CLI/config values before environment and shipped defaults."""
    for source in (overrides, file_data):
        candidate = source.get(section, {})
        if isinstance(candidate, dict) and key in candidate and candidate[key] is not None:
            return candidate[key]
    if environment.get(environment_key):
        return environment[environment_key]
    candidate = default_data.get(section, {})
    return candidate.get(key, fallback) if isinstance(candidate, dict) else fallback


def _environment_values(root: Path) -> dict[str, str]:
    """Load .env values while allowing explicit process variables to take precedence."""
    values = load_dotenv_file(root / ".env")
    for key in ("HEALTH_MODEL_PATH", "INSTRUMENT_MODEL_PATH", "POSE_MODEL_PATH", "SURGICAL_OUTPUT_DIR", "SURGICAL_DEVICE"):
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


def _optional_path(value: Any) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return Path(str(value)).expanduser()


def load_config(project_root: Path | None = None, overrides: dict[str, Any] | None = None, config_path: Path | None = None) -> AppConfig:
    root = (project_root or Path.cwd()).resolve()
    default_data = _read_yaml(root / "configs" / "default.yaml")
    data = deepcopy(default_data)
    file_data: dict[str, Any] = {}
    if config_path is not None:
        resolved_config_path = config_path.expanduser().resolve()
        if not resolved_config_path.is_file():
            raise ValueError(f"Yapılandırma dosyası bulunamadı: {resolved_config_path.name}")
        file_data = _read_yaml(resolved_config_path)
        _merge(data, file_data)
    active_overrides = overrides or {}
    if active_overrides:
        _merge(data, active_overrides)
    env = _environment_values(root)
    runtime = data.get("runtime", {})
    privacy = data.get("privacy", {})
    tracking = data.get("tracking", {})
    depth = data.get("depth", {})
    analytics = data.get("analytics", {})
    model = data.get("model", {})
    cli = data.get("cli", {})
    reproducibility = data.get("reproducibility", {})
    health = tracking.get("health", {})
    instrument = tracking.get("instrument", {})
    names = _read_yaml(root / "configs" / "instrument_names.yaml").get("instrument_display_names", {})
    config = AppConfig(
        output_dir=Path(_source_value(default_data, file_data, active_overrides, "runtime", "output_dir", env, "SURGICAL_OUTPUT_DIR", "outputs")),
        device=str(_source_value(default_data, file_data, active_overrides, "runtime", "device", env, "SURGICAL_DEVICE", "auto")), save_audio=bool(runtime.get("save_audio", True)), max_video_seconds=runtime.get("max_video_seconds"),
        privacy_mode=str(privacy.get("mode", "gaussian")), blur_kernel=int(privacy.get("blur_kernel", 51)), pixelation_factor=int(privacy.get("pixelation_factor", 14)), mask_dilation_px=int(privacy.get("mask_dilation_px", 13)), feather_px=int(privacy.get("feather_px", 9)), persistence_frames=int(privacy.get("persistence_frames", 8)), max_persistence_iou=float(privacy.get("max_persistence_iou", 0.15)), bbox_privacy_fallback=bool(privacy.get("bbox_privacy_fallback", True)),
        tracker_algorithm=str(tracking.get("algorithm", "botsort")), confidence=float(tracking.get("confidence", 0.35)), iou=float(tracking.get("iou", 0.45)), health_confidence=float(health.get("confidence", tracking.get("confidence", 0.35))), health_iou=float(health.get("iou", tracking.get("iou", 0.45))), instrument_confidence=float(instrument.get("confidence", tracking.get("confidence", 0.25))), instrument_iou=float(instrument.get("iou", tracking.get("iou", 0.40))), track_buffer=int(tracking.get("track_buffer", 30)), min_confirm_frames=int(tracking.get("min_confirm_frames", 3)), max_lost_seconds=float(tracking.get("max_lost_seconds", 0.75)),
        max_gap_seconds=float(analytics.get("max_gap_seconds", analytics.get("gap_tolerance_seconds", 0.5))), minimum_interval_seconds=float(analytics.get("minimum_interval_seconds", 0.0)), smoothing_window=int(analytics.get("smoothing_window", 3)), max_relative_jump=float(analytics.get("max_relative_jump", 0.35)),
        depth_enabled=bool(depth.get("enabled", True)), depth_model_id=str(depth.get("model_id", "depth-anything/Depth-Anything-V2-Small-hf")), depth_stride=int(depth.get("depth_stride", depth.get("depth_interval", 3))), depth_device=str(depth.get("device", "auto")),
        fp16=bool(model.get("fp16", True)), image_size=model.get("image_size"), random_seed=reproducibility.get("random_seed", 0), log_level=str(reproducibility.get("log_level", "INFO")), output_codec=str(reproducibility.get("output_codec", "mp4v")), allow_cpu_fallback=bool(model.get("allow_cpu_fallback", False)), health_class_name=model.get("health_class_name"), health_class_id=model.get("health_class_id"),
        camera=data.get("camera", {}),
        health_model_path=_optional_path(_source_value(default_data, file_data, active_overrides, "model", "health_model_path", env, "HEALTH_MODEL_PATH", None)),
        instrument_model_path=_optional_path(_source_value(default_data, file_data, active_overrides, "model", "instrument_model_path", env, "INSTRUMENT_MODEL_PATH", None)),
        pose_model_path=_optional_path(_source_value(default_data, file_data, active_overrides, "model", "pose_model_path", env, "POSE_MODEL_PATH", None)),
        cli_privacy_mode=str(cli.get("privacy_mode", "skeleton-only")),
        instrument_display_names={str(k): str(v) for k, v in names.items()},
    )
    return config.validate()


def write_run_config(path: Path, config: AppConfig) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.public_run_config(), handle, allow_unicode=True, sort_keys=False)
