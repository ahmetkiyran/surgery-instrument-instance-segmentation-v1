"""Human-readable preflight diagnostics for local installations."""

from __future__ import annotations

import importlib
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig
from .depth import DepthEstimator
from .model_loader import inspect_model, resolve_health_class_id
from .model_manager import ModelManager, ModelManagerError
from .utils import hardware_summary, tool_available


@dataclass
class Check:
    label: str
    ok: bool
    detail: str


def _package_check(module: str) -> Check:
    try:
        loaded = importlib.import_module(module)
        return Check(f"Paket: {module}", True, getattr(loaded, "__version__", "yüklü"))
    except Exception as error:
        return Check(f"Paket: {module}", False, str(error))


def run_doctor(project_root: Path, config: AppConfig, health_override: Path | None = None, instrument_override: Path | None = None) -> list[Check]:
    checks = [Check("Python", sys.version_info >= (3, 11), f"{sys.version.split()[0]} ({platform.platform()})")]
    hardware = hardware_summary()
    checks += [Check("PyTorch", hardware["torch"] != "unavailable", str(hardware["torch"])), Check("CUDA", bool(hardware["cuda_available"]), f"{hardware['gpu']} / {hardware['device']}"), Check("FFmpeg", tool_available("ffmpeg"), "ffmpeg PATH üzerinde" if tool_available("ffmpeg") else "FFmpeg bulunamadı")]
    for package in ("ultralytics", "cv2", "gradio", "numpy", "pandas", "plotly", "matplotlib", "yaml", "pydantic", "transformers"):
        checks.append(_package_check(package))
    try:
        output_root = config.output_dir if config.output_dir.is_absolute() else project_root / config.output_dir
        output_root.mkdir(parents=True, exist_ok=True)
        probe = output_root / ".write_probe"
        probe.write_text("ok", encoding="utf-8"); probe.unlink()
        checks.append(Check("Çıktı dizini", True, "Yazılabilir"))
    except Exception as error:
        checks.append(Check("Çıktı dizini", False, str(error)))
    try:
        manager = ModelManager(project_root)
    except ModelManagerError as error:
        manager = None
        checks.append(Check("Model manifest", False, str(error)))
    health_path = health_override or config.health_model_path
    if health_path is None and manager is not None:
        health_status = manager.verify_spec(manager.specs()["health_personnel"])
        health_path = health_status.path if health_status.ready else None
    try:
        if health_path is None:
            detail = health_status.detail if manager is not None else "Health model yolu yok."
            raise RuntimeError(detail)
        health = inspect_model(health_path)
        class_id = resolve_health_class_id(health.names)
        checks.append(Check("Sağlık modeli", True, f"{health.path} | sınıflar: {health.names} | health class ID: {class_id}"))
    except Exception as error:
        checks.append(Check("Sağlık modeli", False, str(error)))
        health = None
    instrument_path = instrument_override or config.instrument_model_path
    if instrument_path is None and manager is not None:
        instrument_status = manager.verify_spec(manager.specs()["surgical_instruments"])
        instrument_path = instrument_status.path if instrument_status.ready else None
    if instrument_path:
        try:
            instrument = inspect_model(instrument_path)
            checks.append(Check("Alet modeli", True, f"{instrument.path} | sınıflar: {instrument.names}"))
        except Exception as error:
            checks.append(Check("Alet modeli", False, str(error)))
    else:
        detail = instrument_status.detail if manager is not None else "Alet modeli yolu yok."
        checks.append(Check("Alet modeli", False, detail + " Yerel .env yolu girin veya models download çalıştırın."))
    depth = DepthEstimator(config.depth_model_id, config.depth_device, config.depth_enabled)
    if not config.depth_enabled:
        checks.append(Check("Derinlik modeli", True, "Kapalı (kullanıcı tercihi)"))
    else:
        try:
            import transformers  # noqa: F401
            cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
            checks.append(Check("Derinlik modeli/cache", True, f"{depth.model_id}; cache: {cache_root} (ilk analizde indirilebilir)"))
        except Exception as error:
            checks.append(Check("Derinlik modeli/cache", False, f"transformers eksik: {error}"))
    return checks


def print_doctor(checks: list[Check]) -> int:
    for check in checks:
        print(f"{'[OK]' if check.ok else '[FAIL]'} {check.label}: {check.detail}")
    return 0 if all(check.ok for check in checks if check.label not in {"CUDA"}) else 2
