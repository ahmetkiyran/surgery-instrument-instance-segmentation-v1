"""Human-readable, non-destructive diagnostics for local CLI readiness."""

from __future__ import annotations

import importlib
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig
from .depth import DepthEstimator
from .model_loader import inspect_model, resolve_health_class_id, validate_model_roles
from .model_manager import ModelManager, ModelManagerError
from .pose_estimator import PoseEstimator
from .utils import hardware_summary, tool_available


@dataclass(frozen=True)
class Check:
    """One readiness result; warnings do not prevent an otherwise safe run."""

    label: str
    ok: bool
    detail: str
    severity: str = "critical"

    @property
    def critical(self) -> bool:
        return self.severity == "critical" and not self.ok


def _package_check(module: str, *, required: bool = True) -> Check:
    try:
        loaded = importlib.import_module(module)
        return Check(f"Paket: {module}", True, str(getattr(loaded, "__version__", "yüklü")))
    except Exception as error:
        return Check(f"Paket: {module}", False, str(error), "critical" if required else "warning")


def _default_pose_path(project_root: Path) -> Path:
    """Use only explicit, non-ambiguous official pose filenames as a fallback."""
    for candidate in (project_root / "models" / "weights" / "yolo11m-pose.pt", project_root / "yolo11m-pose.pt"):
        if candidate.is_file():
            return candidate
    return project_root / "models" / "weights" / "yolo11m-pose.pt"


def _model_paths(
    project_root: Path,
    config: AppConfig,
    health_override: Path | None,
    instrument_override: Path | None,
    pose_override: Path | None,
) -> tuple[Path | None, Path | None, Path]:
    health_path = health_override or config.health_model_path
    instrument_path = instrument_override or config.instrument_model_path
    try:
        manager = ModelManager(project_root)
    except ModelManagerError:
        manager = None
    if manager is not None:
        specs = manager.specs()
        if health_path is None:
            status = manager.verify_spec(specs["health_personnel"])
            health_path = status.path if status.ready else None
        if instrument_path is None:
            status = manager.verify_spec(specs["surgical_instruments"])
            instrument_path = status.path if status.ready else None
    pose_path = pose_override or config.pose_model_path or _default_pose_path(project_root)
    return health_path, instrument_path, pose_path


def _health_checks(path: Path | None, config: AppConfig) -> tuple[list[Check], object | None]:
    if path is None or not path.is_file():
        detail = "Model dosyası bulunamadı." if path is None else f"{path.name} bulunamadı."
        return [
            Check("Health modeli", False, detail),
            Check("Health model görevi", False, "Model doğrulanamadı."),
            Check("Health model sınıfları", False, "Model doğrulanamadı."),
            Check("Health-personnel sınıfı", False, "Model doğrulanamadı."),
        ], None
    try:
        info = inspect_model(path)
        class_id = resolve_health_class_id(info.names, config.health_class_name, config.health_class_id)
        return [
            Check("Health modeli", True, path.name),
            Check("Health model görevi", True, info.task),
            Check("Health model sınıfları", True, str(info.names)),
            Check("Health-personnel sınıfı", True, f"ID {class_id}"),
        ], info
    except Exception as error:
        return [
            Check("Health modeli", False, f"{path.name}: {error}"),
            Check("Health model görevi", False, "Model doğrulanamadı."),
            Check("Health model sınıfları", False, "Model doğrulanamadı."),
            Check("Health-personnel sınıfı", False, "Model doğrulanamadı."),
        ], None


def _instrument_checks(path: Path | None, health_path: Path | None, health_info: object | None, config: AppConfig) -> list[Check]:
    if path is None or not path.is_file():
        detail = "Model dosyası bulunamadı." if path is None else f"{path.name} bulunamadı."
        return [
            Check("Alet modeli", False, detail),
            Check("Alet model görevi", False, "Model doğrulanamadı."),
            Check("Alet model sınıfları", False, "Model doğrulanamadı."),
        ]
    try:
        info = inspect_model(path)
        if health_path is not None and health_info is not None:
            validate_model_roles(health_path, path, health_info, info, config.health_class_name, config.health_class_id)
        return [
            Check("Alet modeli", True, path.name),
            Check("Alet model görevi", True, info.task),
            Check("Alet model sınıfları", True, str(info.names)),
        ]
    except Exception as error:
        return [
            Check("Alet modeli", False, f"{path.name}: {error}"),
            Check("Alet model görevi", False, "Model doğrulanamadı."),
            Check("Alet model sınıfları", False, "Model doğrulanamadı."),
        ]


def _pose_checks(path: Path, config: AppConfig, require_pose: bool) -> list[Check]:
    missing_severity = "critical" if require_pose else "warning"
    if not path.is_file():
        return [
            Check("Pose modeli", False, f"{path.name} bulunamadı.", missing_severity),
            Check("Pose model görevi", False, "Model doğrulanamadı.", missing_severity),
            Check("Pose keypoint sayısı", False, "Model doğrulanamadı.", missing_severity),
        ]
    try:
        info = PoseEstimator(path, config.device, config.fp16).load()
        return [
            Check("Pose modeli", True, path.name),
            Check("Pose model görevi", True, info.task),
            Check("Pose keypoint sayısı", True, str(info.keypoint_shape[0])),
        ]
    except Exception as error:
        return [
            Check("Pose modeli", False, f"{path.name}: {error}", missing_severity),
            Check("Pose model görevi", False, "Model doğrulanamadı.", missing_severity),
            Check("Pose keypoint sayısı", False, "Model doğrulanamadı.", missing_severity),
        ]


def run_doctor(
    project_root: Path,
    config: AppConfig,
    health_override: Path | None = None,
    instrument_override: Path | None = None,
    pose_override: Path | None = None,
    output_override: Path | None = None,
    require_pose: bool = False,
    config_path: Path | None = None,
) -> list[Check]:
    """Check dependencies, writable storage and model metadata without starting analysis."""
    root = project_root.resolve()
    checks = [
        Check("Python", sys.version_info >= (3, 11), f"{sys.version.split()[0]}"),
        Check("İşletim sistemi", True, platform.platform()),
    ]
    hardware = hardware_summary()
    cuda_severity = "critical" if str(config.device).casefold().startswith("cuda") or str(config.device).isdecimal() else "warning"
    checks += [
        Check("PyTorch", hardware["torch"] != "unavailable", str(hardware["torch"])),
        Check("CUDA", bool(hardware["cuda_available"]), f"{hardware['gpu']} / {hardware['device']}", cuda_severity),
        Check("CUDA runtime", hardware["cuda_runtime"] != "unavailable", str(hardware["cuda_runtime"]), cuda_severity),
        Check("GPU adı", hardware["gpu"] != "CPU", str(hardware["gpu"]), cuda_severity),
        Check("FFmpeg", tool_available("ffmpeg"), "PATH üzerinde" if tool_available("ffmpeg") else "Bulunamadı; MP4V fallback kullanılabilir.", "warning"),
        Check("FFprobe", tool_available("ffprobe"), "PATH üzerinde" if tool_available("ffprobe") else "Bulunamadı.", "warning"),
    ]
    for package in ("ultralytics", "cv2", "numpy", "pandas", "plotly", "matplotlib", "yaml", "pydantic"):
        checks.append(_package_check(package))
    checks.append(_package_check("transformers", required=config.depth_enabled))
    if config_path is not None:
        checks.append(Check("Config dosyası", config_path.is_file(), config_path.name if config_path.is_file() else f"{config_path.name} bulunamadı."))
    else:
        checks.append(Check("Config dosyası", True, "Varsayılan yapılandırma"))
    try:
        output_root = output_override or config.output_dir
        output_root = output_root if output_root.is_absolute() else root / output_root
        output_root.mkdir(parents=True, exist_ok=True)
        probe = output_root / ".surgical_pipeline_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks.append(Check("Çıktı dizini", True, "Yazılabilir"))
    except Exception as error:
        checks.append(Check("Çıktı dizini", False, str(error)))

    health_path, instrument_path, pose_path = _model_paths(root, config, health_override, instrument_override, pose_override)
    health_checks, health_info = _health_checks(health_path, config)
    checks.extend(health_checks)
    checks.extend(_instrument_checks(instrument_path, health_path, health_info, config))
    checks.extend(_pose_checks(pose_path, config, require_pose))

    depth = DepthEstimator(config.depth_model_id, config.depth_device, config.depth_enabled)
    if not config.depth_enabled:
        checks.append(Check("Derinlik modeli/cache", True, "Kapalı (kullanıcı tercihi)", "warning"))
    else:
        cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        checks.append(Check("Derinlik modeli/cache", True, f"{depth.model_id}; cache: {cache_root.name}"))
    return checks


def print_doctor(checks: list[Check], *, ascii_only: bool | None = None) -> int:
    """Render a terminal-safe summary and return the documented readiness status."""
    if ascii_only is None:
        encoding = (getattr(sys.stdout, "encoding", "") or "").casefold()
        ascii_only = not encoding.startswith("utf")
    symbols = ("[OK]", "[WARN]", "[FAIL]") if ascii_only else ("✓", "!", "✗")
    successful = warnings = critical = 0
    for check in checks:
        if check.ok:
            successful += 1
            symbol = symbols[0]
        elif check.severity == "warning":
            warnings += 1
            symbol = symbols[1]
        else:
            critical += 1
            symbol = symbols[2]
        print(f"{symbol} {check.label}: {check.detail}")
    print(f"Kontrol: {successful} başarılı, {warnings} uyarı, {critical} kritik hata")
    print("Sistem analize hazır." if critical == 0 else "Sistem analize hazır değil.")
    return 0 if critical == 0 else 4
