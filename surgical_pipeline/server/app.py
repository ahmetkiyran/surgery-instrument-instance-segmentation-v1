"""FastAPI routing only; all analysis remains in the existing Python core."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import cv2

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from .artifacts import Artifact
from .config import ServerSettings
from .jobs import JobManager
from .sessions import SessionStore
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
    SelectionEventRequest,
    SessionResponse,
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
        app.state.sessions = SessionStore()
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

    def sessions(request: Request) -> SessionStore:
        return request.app.state.sessions

    def session_response(item, state: str = "active") -> SessionResponse:
        metadata = item.metadata
        timestamp = item.current_frame / metadata.fps if metadata and metadata.fps else 0.0
        active = [event for event in item.events if event.action in {"add", "select", "resume"}]
        tracks = [{"event_id": event.event_id, "sam3_track_id": event.sam3_track_id, "unified_track_id": event.unified_track_id, "category": event.target_category} for event in active]
        base = f"/api/v1/sessions/{item.session_id}"
        return SessionResponse(session_id=item.session_id, status=state, job_id=item.job_id, current_frame=item.current_frame, current_timestamp=timestamp, video_width=metadata.width if metadata else None, video_height=metadata.height if metadata else None, source_fps=metadata.fps if metadata else None, frame_count=metadata.frame_count if metadata else None, active_tracks=tracks, selected_tracks=[track["unified_track_id"] for track in tracks if track["unified_track_id"]], preview_url=base + "/preview", inspection_preview_url=base + "/preview?overlay=inspection", privacy_preview_url=base + "/preview?overlay=privacy-xray", warnings=item.warnings)

    def _active_events_at(item, frame_index: int):
        return [
            event for event in item.events
            if event.action in {"add", "select", "resume"} and event.frame_index <= frame_index
        ]

    def _draw_selection_markers(frame, item, frame_index: int) -> None:
        """Show recorded prompts before SAM3 propagation has produced masks."""
        if item.metadata is None:
            return
        for event in _active_events_at(item, frame_index):
            x, y = event.pixel(item.metadata.width, item.metadata.height)
            colour = (72, 232, 255) if event.frame_index == frame_index else (120, 190, 220)
            cv2.circle(frame, (x, y), 9, colour, 2, cv2.LINE_AA)
            cv2.line(frame, (x - 14, y), (x + 14, y), colour, 1, cv2.LINE_AA)
            cv2.line(frame, (x, y - 14), (x, y + 14), colour, 1, cv2.LINE_AA)
            label = event.unified_track_id or f"target-{event.sam3_track_id or '?'}"
            cv2.putText(frame, label, (x + 12, max(18, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, .48, colour, 1, cv2.LINE_AA)

    def preview_frame(item, frame_index: int, overlay: str):
        if item.source is None or item.metadata is None:
            raise HTTPException(status_code=404, detail="Oturum videosu bulunamadı.")
        index = min(max(0, frame_index), max(0, item.metadata.frame_count - 1))
        capture = cv2.VideoCapture(str(item.source))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            raise HTTPException(status_code=422, detail="Preview karesi okunamadı.")
        item.current_frame = index
        if overlay == "inspection":
            from ..unified import InspectionRenderer
            frame = InspectionRenderer().render(frame, item.sam3_frames.get(index, []), [], "overlay")
            _draw_selection_markers(frame, item, index)
        elif overlay == "privacy-xray":
            # A live privacy preview is synthetic-only.  Until the pose pass has
            # run, return a black synthetic canvas rather than source RGB.
            frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            frame[:] = (9, 5, 2)
            cv2.putText(frame, "Privacy X-Ray preview is generated during analysis", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, .55, (230, 220, 160), 1, cv2.LINE_AA)
        encoded, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not encoded:
            raise HTTPException(status_code=500, detail="Preview encode edilemedi.")
        return StreamingResponse(iter([data.tobytes()]), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    def privacy_preview_frame(item, frame_index: int):
        """Render a synthetic geometry-only preview without opening source RGB."""
        if item.metadata is None:
            raise HTTPException(status_code=404, detail="Oturum videosu bulunamadÄ±.")
        index = min(max(0, frame_index), max(0, item.metadata.frame_count - 1))
        item.current_frame = index
        from ..unified import PrivacyXRayRenderer
        frame = PrivacyXRayRenderer(settings.project_root).render(
            (item.metadata.width, item.metadata.height), [], [], set(), [], index, item.sam3_frames.get(index, [])
        )
        _draw_selection_markers(frame, item, index)
        if not item.sam3_frames.get(index):
            cv2.putText(frame, "Synthetic privacy-Xray preview", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (210, 220, 230), 1, cv2.LINE_AA)
        encoded, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not encoded:
            raise HTTPException(status_code=500, detail="Privacy preview encode edilemedi.")
        return StreamingResponse(iter([data.tobytes()]), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

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
            if settings.max_upload_duration_seconds is not None and metadata.duration_s > settings.max_upload_duration_seconds:
                raise HTTPException(status_code=413, detail="Video, yapılandırılan maksimum süreyi aşıyor.")
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

    @app.post("/api/v1/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_api_access)])
    async def create_session(store: SessionStore = Depends(sessions)) -> SessionResponse:
        item = store.create()
        return SessionResponse(session_id=item.session_id)

    @app.post("/api/v1/sessions/{session_id}/video", response_model=SessionResponse, dependencies=[Depends(require_api_access)])
    async def attach_session_video(session_id: str, payload: JobCreateRequest, request: Request, store: SessionStore = Depends(sessions)) -> SessionResponse:
        item = store.get(session_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Oturum bulunamadı.")
        try:
            if bool(payload.upload_id) == bool(payload.local_path):
                raise ValueError("Bir upload_id veya yalnızca masaüstü local_path sağlayın.")
            if payload.local_path:
                require_loopback(request)
                source = validate_video_path(Path(payload.local_path))
            else:
                source = resolve_upload(str(payload.upload_id))
            item = store.attach_source(session_id, source)
            return session_response(item, "video_attached")
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/v1/sessions/{session_id}/preview", dependencies=[Depends(require_api_access)])
    async def session_preview(session_id: str, frame_index: int = 0, overlay: str = "clean", store: SessionStore = Depends(sessions)) -> Response:
        item = store.get(session_id)
        if item is None or item.source is None:
            raise HTTPException(status_code=404, detail="Oturum videosu bulunamadı.")
        if overlay not in {"clean", "inspection", "privacy-xray"}:
            raise HTTPException(status_code=422, detail="Geçersiz preview modu.")
        if overlay == "privacy-xray":
            return privacy_preview_frame(item, frame_index)
        return preview_frame(item, frame_index, overlay)

    @app.post("/api/v1/sessions/{session_id}/selections", response_model=SessionResponse, dependencies=[Depends(require_api_access)])
    async def add_selection(session_id: str, payload: SelectionEventRequest, store: SessionStore = Depends(sessions)) -> SessionResponse:
        item = store.get(session_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Oturum bulunamadı.")
        from ..unified import SelectionEvent
        event = SelectionEvent.from_dict(payload.model_dump() | {"session_id": session_id, "created_frame_index": payload.frame_index})
        if event.action in {"add", "select", "resume"} and event.sam3_track_id is None:
            assigned = [int(value.sam3_track_id) for value in item.events if value.sam3_track_id is not None]
            event.sam3_track_id = max(assigned, default=0) + 1
            event.unified_track_id = event.unified_track_id or f"u-sam3-{event.sam3_track_id}"
        item = store.add_event(session_id, event)
        # Selection is deliberately non-blocking.  The SAM3 predictor is a
        # long-running job and is started once, with the complete event list,
        # when the user presses Start tracking.
        return session_response(item, "selection_recorded")

    @app.post("/api/v1/sessions/{session_id}/start", response_model=SessionResponse, dependencies=[Depends(require_api_access)])
    async def start_session(session_id: str, payload: JobCreateRequest, request: Request, jobs: JobManager = Depends(manager), store: SessionStore = Depends(sessions)) -> SessionResponse:
        item = store.get(session_id)
        if item is None or item.source is None:
            raise HTTPException(status_code=404, detail="Oturum veya video bulunamadı.")
        if payload.options.output_directory:
            require_loopback(request)
        active_events = store.active_events(session_id)
        options = payload.options.model_copy(update={"selection_events": [event.__dict__ if hasattr(event, "__dict__") else {name: getattr(event, name) for name in event.__dataclass_fields__} for event in active_events]})
        record = jobs.create(item.source, payload.model_copy(update={"upload_id": None, "local_path": str(item.source), "options": options}))
        item.job_id = record.job_id
        return SessionResponse(session_id=session_id, status="started", job_id=record.job_id)

    @app.get("/api/v1/sessions/{session_id}/status", response_model=SessionResponse, dependencies=[Depends(require_api_access)])
    async def session_status(session_id: str, store: SessionStore = Depends(sessions)) -> SessionResponse:
        item = store.get(session_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Oturum bulunamadı.")
        return session_response(item, "paused" if item.paused else "active")

    @app.get("/api/v1/sessions/{session_id}/events", dependencies=[Depends(require_api_access)])
    async def session_events(session_id: str, store: SessionStore = Depends(sessions)) -> list[dict]:
        item = store.get(session_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Oturum bulunamadı.")
        return [{name: getattr(event, name) for name in event.__dataclass_fields__} for event in item.events]

    @app.post("/api/v1/sessions/{session_id}/selections/{event_id}/remove", response_model=SessionResponse, dependencies=[Depends(require_api_access)])
    async def remove_selection(session_id: str, event_id: str, store: SessionStore = Depends(sessions)) -> SessionResponse:
        item = store.get(session_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Oturum bulunamadı.")
        event = next((value for value in item.events if value.event_id == event_id), None)
        if event is None:
            raise HTTPException(status_code=404, detail="Seçim olayı bulunamadı.")
        event.action = "remove"
        return session_response(item, "selection_removed")

    @app.post("/api/v1/sessions/{session_id}/pause", response_model=SessionResponse, dependencies=[Depends(require_api_access)])
    async def pause_session(session_id: str, request: Request, store: SessionStore = Depends(sessions)) -> SessionResponse:
        item = store.get(session_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Oturum bulunamadı.")
        item.paused = True
        if item.job_id:
            request.app.state.jobs.pause(item.job_id)
        return session_response(item, "paused")

    @app.post("/api/v1/sessions/{session_id}/resume", response_model=SessionResponse, dependencies=[Depends(require_api_access)])
    async def resume_session(session_id: str, request: Request, store: SessionStore = Depends(sessions)) -> SessionResponse:
        item = store.get(session_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Oturum bulunamadı.")
        item.paused = False
        if item.job_id:
            request.app.state.jobs.resume(item.job_id)
        return session_response(item, "active")

    @app.post("/api/v1/sessions/{session_id}/cancel", response_model=SessionResponse, dependencies=[Depends(require_api_access)])
    async def cancel_session(session_id: str, jobs: JobManager = Depends(manager), store: SessionStore = Depends(sessions)) -> SessionResponse:
        item = store.get(session_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Oturum bulunamadı.")
        if item.job_id:
            jobs.cancel(item.job_id)
        return SessionResponse(session_id=session_id, status="cancel_requested", job_id=item.job_id)

    @app.get("/api/v1/sessions/{session_id}/artifacts", response_model=list[ArtifactResponse], dependencies=[Depends(require_api_access)])
    async def session_artifacts(session_id: str, jobs: JobManager = Depends(manager), store: SessionStore = Depends(sessions)) -> list[ArtifactResponse]:
        item = store.get(session_id)
        artifacts = jobs.artifacts(item.job_id) if item and item.job_id else None
        return [artifact.response() for artifact in artifacts.values()] if artifacts else []

    @app.get("/api/v1/sessions/{session_id}/tracks", dependencies=[Depends(require_api_access)])
    async def session_tracks(session_id: str, jobs: JobManager = Depends(manager), store: SessionStore = Depends(sessions)) -> dict:
        item = store.get(session_id)
        record = jobs.get(item.job_id) if item and item.job_id else None
        if record is None:
            return {"tracks": []}
        track_file = record.job_root / "unified_tracks.csv"
        if not track_file.is_file():
            return {"tracks": []}
        import csv
        with track_file.open(encoding="utf-8", newline="") as handle:
            return {"tracks": list(csv.DictReader(handle))}

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

    @app.post("/api/v1/jobs/{job_id}/pause", response_model=JobResponse, dependencies=[Depends(require_api_access)])
    async def pause_job(job_id: str, jobs: JobManager = Depends(manager)) -> JobResponse:
        record = jobs.pause(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Ä°ÅŸ bulunamadÄ±.")
        return record.response()

    @app.post("/api/v1/jobs/{job_id}/resume", response_model=JobResponse, dependencies=[Depends(require_api_access)])
    async def resume_job(job_id: str, jobs: JobManager = Depends(manager)) -> JobResponse:
        record = jobs.resume(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Ä°ÅŸ bulunamadÄ±.")
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
