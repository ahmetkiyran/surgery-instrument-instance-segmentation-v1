"""Model discovery, class resolution and isolated YOLO loading."""

from __future__ import annotations

from datetime import UTC, datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .schemas import ModelInfo
from .utils import sha256_file

HEALTH_CLASS_VARIANTS = {
    "health_personel",
    "health_personnel",
    "health_person",
    "healthcare_personnel",
}


class ModelValidationError(RuntimeError):
    """A model is missing, unreadable or does not have the required semantic class."""


@dataclass(frozen=True)
class ModelCandidate:
    path: Path
    size_bytes: int
    modified_utc: str
    names: dict[int, str] | None
    error: str | None = None

    def display(self) -> dict[str, object]:
        return {"path": str(self.path), "size_bytes": self.size_bytes, "modified_utc": self.modified_utc, "classes": self.names, "error": self.error}


def normalize_names(names: object) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    raise ModelValidationError("Modelin model.names alanı okunamadı.")


def resolve_health_class_id(names: dict[int, str]) -> int:
    """Resolve only explicit supported health-personnel labels; never assume class 0."""
    matched = [class_id for class_id, name in names.items() if name.strip().casefold() in HEALTH_CLASS_VARIANTS]
    if len(matched) == 1:
        return matched[0]
    available = ", ".join(f"{key}: {value}" for key, value in names.items())
    if len(matched) > 1:
        raise ModelValidationError(f"Birden fazla sağlık personeli sınıfı eşleşti: {matched}. Model sınıfları: {available}")
    raise ModelValidationError(
        "Sağlık personeli sınıfı bulunamadı. Desteklenen adlar: "
        f"{', '.join(sorted(HEALTH_CLASS_VARIANTS))}. Model sınıfları: {available}"
    )


def _yolo(path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ModelValidationError("Ultralytics kurulu değil. requirements.txt dosyasını kurun.") from error
    try:
        return YOLO(str(path))
    except Exception as error:
        raise ModelValidationError(f"Model açılamadı: {path.name}: {error}") from error


def inspect_model(path: Path) -> ModelInfo:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ModelValidationError(f"Model dosyası bulunamadı: {path}")
    model = _yolo(path)
    names = normalize_names(model.names)
    return ModelInfo(path=str(path), sha256=sha256_file(path), names=names, task=str(getattr(model, "task", "unknown")))


def read_candidate(path: Path) -> ModelCandidate:
    stat = path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
    try:
        return ModelCandidate(path.resolve(), stat.st_size, modified, inspect_model(path).names)
    except Exception as error:
        return ModelCandidate(path.resolve(), stat.st_size, modified, None, str(error))


def discover_instrument_candidates(project_root: Path) -> list[ModelCandidate]:
    """Inspect every best.pt below surgery_training/runs_correct, in a deterministic order."""
    root = project_root / "surgery_training" / "runs_correct"
    if not root.is_dir():
        return []
    paths = sorted(root.rglob("best.pt"), key=lambda item: str(item).casefold())
    return [read_candidate(path) for path in paths if path.is_file()]


def choose_instrument_candidate(candidates: Iterable[ModelCandidate], health_names: dict[int, str]) -> Path:
    candidates = list(candidates)
    viable = [candidate for candidate in candidates if candidate.names and candidate.names != health_names and not any(name.casefold() in HEALTH_CLASS_VARIANTS for name in candidate.names.values())]
    if len(viable) == 1:
        return viable[0].path
    details = "\n".join(str(candidate.display()) for candidate in candidates) or "Aday bulunamadı."
    if not viable:
        raise ModelValidationError("Alet sınıfları içeren kesin bir model adayı bulunamadı. Adaylar:\n" + details)
    raise ModelValidationError("Birden fazla alet modeli adayı var; .env içinde INSTRUMENT_MODEL_PATH seçin. Adaylar:\n" + details)


def resolve_model_paths(
    project_root: Path,
    health_path: Path | None,
    instrument_path: Path | None,
    auto_download: bool = True,
) -> tuple[Path, Path, int, list[ModelCandidate]]:
    """Resolve paths by CLI/.env, verified release weights, then an authorised release download."""
    if health_path is None:
        from .model_manager import release_model_path

        health = release_model_path(project_root, "health_personnel", auto_download)
    else:
        health = health_path.expanduser()
    health_info = inspect_model(health)
    health_id = resolve_health_class_id(health_info.names)
    candidates = discover_instrument_candidates(project_root)
    if instrument_path:
        instrument = instrument_path.expanduser()
        instrument_info = inspect_model(instrument)
        if instrument_info.names == health_info.names or any(name.casefold() in HEALTH_CLASS_VARIANTS for name in instrument_info.names.values()):
            raise ModelValidationError("Seçilen alet modeli sağlık modeliyle aynı role sahip görünüyor; sınıflar: " + str(instrument_info.names))
    else:
        from .model_manager import release_model_path

        instrument = release_model_path(project_root, "surgical_instruments", auto_download)
        instrument_info = inspect_model(instrument)
        if instrument_info.names == health_info.names or any(name.casefold() in HEALTH_CLASS_VARIANTS for name in instrument_info.names.values()):
            raise ModelValidationError("Release alet modeli sağlık modeliyle aynı role sahip görünüyor.")
    return health.resolve(), instrument.resolve(), health_id, candidates


def load_yolo_models(health_path: Path, instrument_path: Path):
    """Load models exactly once per analysis; their tracker states remain independent."""
    return _yolo(health_path), _yolo(instrument_path)
