"""Secure, manifest-driven GitHub Release model acquisition."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from .model_loader import ModelValidationError, inspect_model, resolve_health_class_id


class ModelManagerError(RuntimeError):
    """A release asset cannot be safely obtained or is not the expected model."""


DownloadProgress = Callable[[str, int, int | None], None]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str
    download_url: str
    sha256: str
    task: str
    expected_classes: tuple[str, ...]

    def has_download_url(self) -> bool:
        parsed = urlparse(self.download_url)
        return parsed.scheme in {"http", "https"} and "GITHUB_RELEASE_DOWNLOAD_URL" not in self.download_url


@dataclass(frozen=True)
class ModelStatus:
    key: str
    path: Path
    state: str
    detail: str

    @property
    def ready(self) -> bool:
        return self.state == "ready"


class ModelManager:
    """Checks, downloads and verifies only manifest-defined release assets."""

    def __init__(self, project_root: Path, manifest_path: Path | None = None, weights_dir: Path | None = None) -> None:
        self.project_root = project_root.resolve()
        self.manifest_path = manifest_path or self.project_root / "models" / "model_manifest.json"
        self.weights_dir = weights_dir or self.project_root / "models" / "weights"
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        if not self.manifest_path.is_file():
            raise ModelManagerError(f"Model manifest bulunamadı: {self.manifest_path}")
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ModelManagerError("Model manifest geçerli JSON değil.") from error
        if not isinstance(data.get("models"), dict):
            raise ModelManagerError("Model manifest 'models' nesnesini içermelidir.")
        return data

    @property
    def release_published(self) -> bool:
        return bool(self._manifest.get("release_published", False))

    def specs(self) -> dict[str, ModelSpec]:
        result: dict[str, ModelSpec] = {}
        for key, values in self._manifest["models"].items():
            try:
                result[key] = ModelSpec(
                    key=key,
                    filename=str(values["filename"]),
                    download_url=str(values["download_url"]),
                    sha256=str(values["sha256"]).lower(),
                    task=str(values["task"]),
                    expected_classes=tuple(str(item) for item in values["expected_classes"]),
                )
            except (KeyError, TypeError) as error:
                raise ModelManagerError(f"Model manifest geçersiz: {key}") from error
        return result

    def local_path(self, spec: ModelSpec) -> Path:
        candidate = (self.weights_dir / spec.filename).resolve()
        root = self.weights_dir.resolve()
        if candidate.parent != root or candidate.suffix.lower() != ".pt":
            raise ModelManagerError(f"Güvenli olmayan model dosya adı: {spec.filename}")
        return candidate

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def verify_spec(self, spec: ModelSpec) -> ModelStatus:
        path = self.local_path(spec)
        part = path.with_suffix(path.suffix + ".part")
        if not path.is_file():
            if part.exists():
                return ModelStatus(spec.key, path, "partial", "Yarım indirme bulundu (.part); download veya redownload ile tamamlayın.")
            return ModelStatus(spec.key, path, "missing", "Model dosyası bulunamadı.")
        actual_hash = self._sha256(path)
        if actual_hash != spec.sha256:
            return ModelStatus(spec.key, path, "invalid_hash", "SHA-256 eşleşmiyor; bu dosya kullanılmayacak.")
        try:
            info = inspect_model(path)
            actual_classes = tuple(info.names[index] for index in sorted(info.names))
            if info.task != spec.task:
                return ModelStatus(spec.key, path, "invalid_model", f"Task uyuşmuyor: {info.task} (beklenen {spec.task})")
            if actual_classes != spec.expected_classes:
                return ModelStatus(spec.key, path, "invalid_model", f"Sınıflar uyuşmuyor: {actual_classes}")
            if spec.key == "health_personnel":
                resolve_health_class_id(info.names)
        except (ModelValidationError, OSError, ValueError) as error:
            return ModelStatus(spec.key, path, "invalid_model", str(error))
        return ModelStatus(spec.key, path, "ready", "SHA-256, task ve model sınıfları doğrulandı.")

    def statuses(self) -> list[ModelStatus]:
        return [self.verify_spec(spec) for spec in self.specs().values()]

    def release_paths_if_ready(self) -> tuple[Path, Path] | None:
        statuses = {status.key: status for status in self.statuses()}
        if not all(status.ready for status in statuses.values()):
            return None
        return statuses["health_personnel"].path, statuses["surgical_instruments"].path

    def _download(self, spec: ModelSpec, force: bool, progress: DownloadProgress | None) -> Path:
        path = self.local_path(spec)
        status = self.verify_spec(spec)
        if status.ready and not force:
            return path
        if not self.release_published or not spec.has_download_url():
            raise ModelManagerError(
                "Model release URL'si henüz yayınlanmadı. Manuel indirme için README'deki release talimatını kullanın "
                "veya yerel .env içinde model yollarını belirtin."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(path.suffix + ".part")
        if part.exists() and progress:
            progress(spec.key, 0, None)
        request = urllib.request.Request(spec.download_url, headers={"User-Agent": "surgical-video-analytics/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=45) as response, part.open("wb") as output:
                total = response.headers.get("Content-Length")
                total_bytes = int(total) if total and total.isdigit() else None
                current = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    current += len(chunk)
                    if progress:
                        progress(spec.key, current, total_bytes)
        except (urllib.error.URLError, OSError) as error:
            raise ModelManagerError(f"İndirme bağlantısı başarısız. Manuel bağlantı: {spec.download_url}. Ayrıntı: {error}") from error
        if self._sha256(part) != spec.sha256:
            part.unlink(missing_ok=True)
            raise ModelManagerError("İndirilen modelin SHA-256 değeri eşleşmedi; bozuk dosya kullanılmadı.")
        # Atomic replacement occurs only after hash success; keep any previous file recoverable while validation runs.
        backup = path.with_suffix(path.suffix + ".previous")
        backup.unlink(missing_ok=True)
        if path.exists():
            os.replace(path, backup)
        os.replace(part, path)
        verified = self.verify_spec(spec)
        if not verified.ready:
            path.unlink(missing_ok=True)
            if backup.exists():
                os.replace(backup, path)
            raise ModelManagerError(f"İndirilen dosya model doğrulamasını geçemedi: {verified.detail}")
        backup.unlink(missing_ok=True)
        return path

    def ensure(self, key: str, force: bool = False, progress: DownloadProgress | None = None) -> Path:
        specs = self.specs()
        if key not in specs:
            raise ModelManagerError(f"Manifest içinde tanımsız model: {key}")
        return self._download(specs[key], force, progress)

    def download_all(self, force: bool = False, progress: DownloadProgress | None = None) -> tuple[Path, Path]:
        specs = self.specs()
        health = self._download(specs["health_personnel"], force, progress)
        instrument = self._download(specs["surgical_instruments"], force, progress)
        return health, instrument


def release_model_paths(project_root: Path, auto_download: bool = True) -> tuple[Path, Path]:
    """Resolve verified release assets, downloading them only when a real release is published."""
    manager = ModelManager(project_root)
    paths = manager.release_paths_if_ready()
    if paths:
        return paths
    if not auto_download:
        raise ModelManagerError("Release modelleri yerelde hazır değil.")
    return manager.download_all()


def release_model_path(project_root: Path, key: str, auto_download: bool = True) -> Path:
    """Resolve one verified release model for priority-aware model selection."""
    manager = ModelManager(project_root)
    spec = manager.specs().get(key)
    if spec is None:
        raise ModelManagerError(f"Manifest içinde tanımsız model: {key}")
    status = manager.verify_spec(spec)
    if status.ready:
        return status.path
    if not auto_download:
        raise ModelManagerError(status.detail)
    return manager.ensure(key)
