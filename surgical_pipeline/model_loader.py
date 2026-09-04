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


def _normalise_class_name(name: str) -> str:
    return name.strip().casefold().replace("-", "_").replace(" ", "_")


def resolve_health_class_id(names: dict[int, str], name_override: str | None = None, id_override: int | None = None) -> int:
    """Resolve a personnel class from metadata, never from an assumed class position."""
    if name_override is not None and id_override is not None:
        raise ModelValidationError("Health class için aynı anda isim ve ID override verilemez.")
    if id_override is not None:
        if id_override not in names:
            raise ModelValidationError(f"Health class ID {id_override} model metadata'sında yok: {names}")
        return id_override
    if name_override is not None:
        matched = [class_id for class_id, name in names.items() if _normalise_class_name(name) == _normalise_class_name(name_override)]
        if len(matched) == 1:
            return matched[0]
        if not matched:
            raise ModelValidationError(f"Health class '{name_override}' model metadata'sında yok: {names}")
        raise ModelValidationError(f"Health class adı '{name_override}' belirsiz: {matched}")
    matched = [class_id for class_id, name in names.items() if _normalise_class_name(name) in HEALTH_CLASS_VARIANTS]
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
    return inspect_loaded_model(path, model)


def inspect_loaded_model(path: Path, model) -> ModelInfo:
    """Read metadata from an already-open model without loading weights again."""
    path = path.expanduser().resolve()
    names = normalize_names(model.names)
    task = str(getattr(model, "task", "unknown"))
    if task != "segment":
        raise ModelValidationError(f"Model instance segmentation göreviyle yüklenmedi (task={task}).")
    if not names:
        raise ModelValidationError("Modelin model.names alanı boş.")
    return ModelInfo(path=str(path), sha256=sha256_file(path), names=names, task=task)


def load_validated_yolo_models(
    health_path: Path,
    instrument_path: Path,
    health_name_override: str | None = None,
    health_id_override: int | None = None,
):
    """Load exactly one YOLO object per role and validate their runtime metadata."""
    health_model, instrument_model = _yolo(health_path), _yolo(instrument_path)
    health_info = inspect_loaded_model(health_path, health_model)
    instrument_info = inspect_loaded_model(instrument_path, instrument_model)
    health_id = validate_model_roles(health_path, instrument_path, health_info, instrument_info, health_name_override, health_id_override)
    return health_model, instrument_model, health_info, instrument_info, health_id


def resolve_model_file_paths(project_root: Path, health_path: Path | None, instrument_path: Path | None, auto_download: bool = True) -> tuple[Path, Path]:
    """Resolve only files; semantic validation occurs after the single runtime load."""
    if health_path is None:
        from .model_manager import release_model_path

        health = release_model_path(project_root, "health_personnel", auto_download)
    else:
        health = health_path.expanduser()
    if instrument_path is None:
        from .model_manager import release_model_path

        instrument = release_model_path(project_root, "surgical_instruments", auto_download)
    else:
        instrument = instrument_path.expanduser()
    health, instrument = health.resolve(), instrument.resolve()
    if not health.is_file() or not instrument.is_file():
        missing = health if not health.is_file() else instrument
        raise ModelValidationError(f"Model dosyası bulunamadı: {missing}")
    if health == instrument:
        raise ModelValidationError("Aynı model dosyası health ve instrument rolleri için kullanılamaz.")
    return health, instrument


def validate_model_roles(
    health_path: Path,
    instrument_path: Path,
    health_info: ModelInfo,
    instrument_info: ModelInfo,
    health_name_override: str | None = None,
    health_id_override: int | None = None,
) -> int:
    """Reject swapped/reused weights before any inference begins."""
    try:
        same_file = health_path.resolve() == instrument_path.resolve()
    except OSError:
        same_file = str(health_path) == str(instrument_path)
    if same_file or health_info.sha256 == instrument_info.sha256:
        raise ModelValidationError("Aynı model dosyası health ve instrument rolleri için kullanılamaz.")
    health_id = resolve_health_class_id(health_info.names, health_name_override, health_id_override)
    instrument_health_classes = [name for name in instrument_info.names.values() if _normalise_class_name(name) in HEALTH_CLASS_VARIANTS]
    if instrument_health_classes:
        raise ModelValidationError("Alet modeli health-personnel sınıfı içeriyor; model rolleri ters veya belirsiz: " + str(instrument_info.names))
    if not instrument_info.names:
        raise ModelValidationError("Alet modelinde kullanılabilir sınıf yok.")
    return health_id


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
    health_name_override: str | None = None,
    health_id_override: int | None = None,
) -> tuple[Path, Path, int, list[ModelCandidate]]:
    """Resolve paths by CLI/.env, verified release weights, then an authorised release download."""
    if health_path is None:
        from .model_manager import release_model_path

        health = release_model_path(project_root, "health_personnel", auto_download)
    else:
        health = health_path.expanduser()
    health_info = inspect_model(health)
    health_id = resolve_health_class_id(health_info.names, health_name_override, health_id_override)
    candidates = discover_instrument_candidates(project_root)
    if instrument_path:
        instrument = instrument_path.expanduser()
        instrument_info = inspect_model(instrument)
    else:
        from .model_manager import release_model_path

        instrument = release_model_path(project_root, "surgical_instruments", auto_download)
        instrument_info = inspect_model(instrument)
    health, instrument = health.resolve(), instrument.resolve()
    health_id = validate_model_roles(health, instrument, health_info, instrument_info, health_name_override, health_id_override)
    return health, instrument, health_id, candidates


def load_yolo_models(health_path: Path, instrument_path: Path):
    """Load models exactly once per analysis; their tracker states remain independent."""
    return _yolo(health_path), _yolo(instrument_path)
