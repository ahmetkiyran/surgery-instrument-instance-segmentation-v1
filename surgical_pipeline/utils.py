"""Small, security-conscious utility helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUN_NAME_RE = re.compile(r"^run_\d{8}_\d{6}(?:_\d+)?$")


def configure_logging(log_path: Path | None = None, level: str = "INFO") -> logging.Logger:
    """Return the package logger without logging private input paths to reports."""
    logger = logging.getLogger("surgical_pipeline")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(stream)
    if log_path and not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path for h in logger.handlers):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    return logger


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_run_name(now: datetime | None = None) -> str:
    return "run_" + (now or datetime.now(UTC)).strftime("%Y%m%d_%H%M%S")


def safe_run_dir(output_root: Path) -> Path:
    """Create a unique run directory under *output_root*, rejecting traversal."""
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        name = utc_run_name() if index == 0 else f"{utc_run_name()}_{index}"
        candidate = (root / name).resolve()
        if candidate.parent != root or not RUN_NAME_RE.match(candidate.name):
            raise ValueError("Güvenli olmayan çıktı dizini oluşturma denemesi.")
        try:
            candidate.mkdir()
            (candidate / "logs").mkdir()
            return candidate
        except FileExistsError:
            index += 1


def json_safe(value: Any) -> Any:
    """Convert numeric/container values to strict JSON without NaN or Infinity."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")


def tool_available(name: str) -> bool:
    return shutil.which(name) is not None


def load_dotenv_file(path: Path) -> dict[str, str]:
    """Read only uncomplicated KEY=VALUE lines; avoids an extra dependency at startup."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def scrub_filename(path: Path) -> str:
    """A report may identify the file type but must not expose a patient's filename."""
    return f"input{path.suffix.lower() or '.video'}"


def hardware_summary() -> dict[str, str | bool]:
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        return {
            "device": "cuda:0" if cuda else "cpu",
            "cuda_available": cuda,
            "cuda_runtime": str(torch.version.cuda or "unavailable"),
            "gpu": torch.cuda.get_device_name(0) if cuda else "CPU",
            "torch": str(torch.__version__),
        }
    except Exception:
        return {"device": "cpu", "cuda_available": False, "cuda_runtime": "unavailable", "gpu": "CPU", "torch": "unavailable"}
