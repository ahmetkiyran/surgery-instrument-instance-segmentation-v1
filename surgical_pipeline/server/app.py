"""FastAPI routing only; all analysis remains in the existing Python core."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .artifacts import Artifact
from .config import ServerSettings
from .jobs import JobManager
from .schemas import (
    ArtifactResponse,
    DefaultsResponse,
    DoctorCheckResponse,
    DoctorResponse,
    HealthResponse,
    JobCreateRequest,
    JobResponse,
    LocalVideoInspectionRequest,
    LocalVideoInspectionResponse,
    ModelResponse,
    UploadResponse,
)
from .security import require_api_access, require_loopback, sanitize_filename, validate_video_path


def create_app(settings: ServerSettings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings.uploads_root.mkdir(parents=True, exist_ok=True)
        settings.jobs_root.mkdir(parents=True, exist_ok=True)
        app.state.settings = settings
        app.state.jobs = JobManager(settings)
        yield

    app = FastAPI(title="Surgical Pipeline Local API", version="1.0.0", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-Api-Token", "X-File-Name", "X-Sensitive-Artifact-Confirmed"],
    )

    def manager(request: Request) -> JobManager:
        return request.app.state.jobs

    @app.get("/api/v1/health", response_model=HealthResponse, dependencies=[Depends(require_api_access)])
    async def health(request: Request) -> HealthResponse:
        return HealthResponse(auth_required=bool(settings.token), lan_mode=settings.lan_mode)

    @app.get("/api/v1/system/doctor", response_model=DoctorResponse, dependencies=[Depends(require_api_access)])
    async def doctor() -> DoctorResponse:
        from ..config import load_config
        from ..doctor import run_doctor

        checks = run_doctor(settings.project_root, load_config(settings.project_root))
        public_checks = [
            DoctorCheckResponse(
                label=check.label,
                ok=check.ok,
                detail=check.detail if check.ok else "Bu kontrol tamamlanamadı; yerel Doctor çıktısını inceleyin.",
                severity=check.severity,
            )
            for check in checks
        ]
        return DoctorResponse(ready=not any(not item.ok and item.severity == "critical" for item in public_checks), checks=public_checks)

    @app.get("/api/v1/models", response_model=list[ModelResponse], dependencies=[Depends(require_api_access)])
    async def models() -> list[ModelResponse]:
        from ..cli import _default_pose_path
        from ..model_manager import ModelManager, ModelManagerError

        result: list[ModelResponse] = []
        try:
            statuses = ModelManager(settings.project_root).statuses()
            result.extend(ModelResponse(key=item.key, state=item.state, detail=item.detail, ready=item.ready) for item in statuses)
        except ModelManagerError:
            result.extend([
                ModelResponse(key="health_personnel", state="unavailable", detail="Model manifesti okunamadı.", ready=False),
                ModelResponse(key="surgical_instruments", state="unavailable", detail="Model manifesti okunamadı.", ready=False),
            ])
        pose = _default_pose_path(settings.project_root)
        result.append(ModelResponse(key="pose", state="ready" if pose.is_file() else "missing", detail="Yerel pose modeli" if pose.is_file() else "Pose modeli bulunamadı.", ready=pose.is_file()))
        return result

    @app.get("/api/v1/config/defaults", response_model=DefaultsResponse, dependencies=[Depends(require_api_access)])
    async def defaults() -> DefaultsResponse:
        from ..config import load_config

        config = load_config(settings.project_root)
        return DefaultsResponse(
            privacy_mode=config.cli_privacy_mode,  # type: ignore[arg-type]
            device=config.device,
            depth_enabled=config.depth_enabled,
            health_confidence=config.health_confidence,
            instrument_confidence=config.instrument_confidence,
            iou=config.iou,
            tracker=config.tracker_algorithm,  # type: ignore[arg-type]
        )

    @app.post("/api/v1/uploads", response_model=UploadResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_api_access)])
    async def upload(request: Request) -> UploadResponse:
        from uuid import uuid4
        from ..video_io import VideoError, read_metadata

        filename = sanitize_filename(request.headers.get("x-file-name"))
        if Path(filename).suffix.casefold() not in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
            raise HTTPException(status_code=415, detail="Desteklenen bir video dosyası yükleyin.")
        upload_id = str(uuid4())
        part = settings.uploads_root / f"{upload_id}.part"
        target = settings.uploads_root / f"{upload_id}{Path(filename).suffix.casefold()}"
        total = 0
        try:
            with part.open("wb") as output:
                async for chunk in request.stream():
                    total += len(chunk)
                    if total > settings.max_upload_bytes:
                        raise HTTPException(status_code=413, detail="Dosya, yapılandırılan upload sınırını aşıyor.")
                    output.write(chunk)
            part.replace(target)
            metadata = read_metadata(target)
        except HTTPException:
            part.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            raise
        except (OSError, VideoError):
            part.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail="Yüklenen dosya açılabilir bir video değil.")
        return UploadResponse(upload_id=upload_id, filename=filename, size_bytes=total, duration_seconds=metadata.duration_s)

    @app.post(
        "/api/v1/videos/inspect",
        response_model=LocalVideoInspectionResponse,
        dependencies=[Depends(require_api_access)],
    )
    async def inspect_local_video(payload: LocalVideoInspectionRequest, request: Request) -> LocalVideoInspectionResponse:
        """Read local metadata for the desktop picker without persisting or returning its path."""
        from ..video_io import VideoError, read_metadata

        require_loopback(request)
        try:
            source = validate_video_path(Path(payload.local_path))
            metadata = read_metadata(source)
        except (OSError, ValueError, VideoError) as error:
            raise HTTPException(status_code=422, detail="Desteklenen, aÃ§Ä±labilir bir video dosyasÄ± seÃ§in.") from error
        return LocalVideoInspectionResponse(
            filename=source.name,
            size_bytes=source.stat().st_size,
            duration_seconds=metadata.duration_s,
        )

    def resolve_upload(upload_id: str) -> Path:
        matches = list(settings.uploads_root.glob(f"{upload_id}.*"))
        candidates = [item for item in matches if item.is_file() and item.suffix != ".part"]
        if len(candidates) != 1:
            raise ValueError("Yükleme bulunamadı veya artık kullanılamıyor.")
        return validate_video_path(candidates[0])

    @app.post("/api/v1/jobs", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_api_access)])
    async def create_job(payload: JobCreateRequest, request: Request, jobs: JobManager = Depends(manager)) -> JobResponse:
        if bool(payload.upload_id) == bool(payload.local_path):
            raise HTTPException(status_code=422, detail="Bir upload_id veya yalnızca masaüstü için local_path sağlayın.")
        try:
            if payload.options.output_directory:
                require_loopback(request)
            if payload.local_path:
                require_loopback(request)
                source = validate_video_path(Path(payload.local_path))
                if payload.options.output_directory is None:
                    raise ValueError("Yerel masaüstü yolu için bir çıktı dizini seçin.")
            else:
                source = resolve_upload(str(payload.upload_id))
            record = jobs.create(source, payload)
            return record.response()
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/v1/jobs", response_model=list[JobResponse], dependencies=[Depends(require_api_access)])
    async def list_jobs(jobs: JobManager = Depends(manager)) -> list[JobResponse]:
        return [item.response() for item in jobs.list()]

    @app.get("/api/v1/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(require_api_access)])
    async def get_job(job_id: str, jobs: JobManager = Depends(manager)) -> JobResponse:
        record = jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="İş bulunamadı.")
        return record.response()

    @app.post("/api/v1/jobs/{job_id}/cancel", response_model=JobResponse, dependencies=[Depends(require_api_access)])
    async def cancel_job(job_id: str, jobs: JobManager = Depends(manager)) -> JobResponse:
        record = jobs.cancel(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="İş bulunamadı.")
        return record.response()

    @app.get("/api/v1/jobs/{job_id}/artifacts", response_model=list[ArtifactResponse], dependencies=[Depends(require_api_access)])
    async def list_artifacts(job_id: str, jobs: JobManager = Depends(manager)) -> list[ArtifactResponse]:
        artifacts = jobs.artifacts(job_id)
        if artifacts is None:
            raise HTTPException(status_code=404, detail="İş bulunamadı.")
        return [item.response() for item in artifacts.values()]

    @app.get("/api/v1/jobs/{job_id}/artifacts/{artifact_id}", dependencies=[Depends(require_api_access)])
    async def download_artifact(job_id: str, artifact_id: str, request: Request, jobs: JobManager = Depends(manager)) -> Response:
        artifacts = jobs.artifacts(job_id)
        artifact: Artifact | None = artifacts.get(artifact_id) if artifacts else None
        if artifact is None or not artifact.path.is_file():
            raise HTTPException(status_code=404, detail="Artifact bulunamadı.")
        if artifact.sensitive and request.headers.get("x-sensitive-artifact-confirmed", "").casefold() != "true":
            raise HTTPException(status_code=428, detail="Blur çıktısı hassas içerik olabilir; açık onay gereklidir.")
        return FileResponse(artifact.path, media_type=artifact.media_type, filename=artifact.filename)

    return app
