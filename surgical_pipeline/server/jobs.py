"""Single-worker orchestration around the existing Python pipeline APIs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
import threading
import time
from pathlib import Path
from uuid import uuid4

from .artifacts import Artifact, collect_artifacts
from .config import ServerSettings
from .schemas import JobCreateRequest, JobResponse
from .security import validate_video_path


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class JobRecord:
    job_id: str
    source_path: Path
    request: JobCreateRequest
    job_root: Path
    status: str = "queued"
    progress_percent: float = 0.0
    processed_frames: int = 0
    total_frames: int = 0
    estimated_remaining_seconds: float | None = None
    current_stage: str = "Kuyrukta"
    message: str = "Analiz sırasını bekliyor."
    created_at: datetime = field(default_factory=_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_code: str | None = None
    safe_error_message: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    pause_event: threading.Event = field(default_factory=threading.Event, repr=False)
    artifacts: dict[str, Artifact] = field(default_factory=dict, repr=False)

    def response(self) -> JobResponse:
        elapsed = (time.time() - self.started_at.timestamp()) if self.started_at else 0.0
        return JobResponse(
            job_id=self.job_id,
            status=self.status,  # type: ignore[arg-type]
            progress_percent=round(self.progress_percent, 1),
            processed_frames=self.processed_frames,
            total_frames=self.total_frames,
            elapsed_seconds=max(0.0, elapsed),
            estimated_remaining_seconds=self.estimated_remaining_seconds,
            current_stage=self.current_stage,
            message=self.message,
            created_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            privacy_mode=self.request.options.privacy_mode,
            artifacts=[item.response() for item in self.artifacts.values()],
            error_code=self.error_code,
            safe_error_message=self.safe_error_message,
        )


class JobManager:
    """One daemon worker guarantees that two GPU-heavy inferences never overlap."""

    def __init__(self, settings: ServerSettings) -> None:
        self.settings = settings
        self._jobs: dict[str, JobRecord] = {}
        self._queue: deque[str] = deque()
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._worker = threading.Thread(target=self._worker_loop, name="surgical-analysis-worker", daemon=True)
        self._worker.start()

    def create(self, source_path: Path, request: JobCreateRequest) -> JobRecord:
        with self._wake:
            job_id = str(uuid4())
            output_root = self._output_root(request)
            job_root = output_root / f"job_{job_id}"
            job_root.mkdir(parents=True, exist_ok=False)
            record = JobRecord(job_id=job_id, source_path=source_path, request=request, job_root=job_root)
            self._jobs[job_id] = record
            self._queue.append(job_id)
            self._wake.notify()
            return record

    def _output_root(self, request: JobCreateRequest) -> Path:
        configured = request.options.output_directory
        if configured:
            candidate = Path(configured).expanduser().resolve(strict=False)
            if not candidate.is_absolute() or candidate.suffix:
                raise ValueError("Çıktı dizini geçerli bir klasör olmalıdır.")
            return candidate
        return self.settings.jobs_root

    def list(self) -> list[JobRecord]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> JobRecord | None:
        with self._wake:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            if record.status in {"completed", "failed", "cancelled"}:
                return record
            record.cancel_event.set()
            record.message = "Güvenli iptal isteği kaydedildi; etkin kare sınırında duracak."
            if record.status == "queued":
                record.status = "cancelled"
                record.current_stage = "İptal edildi"
                record.finished_at = _now()
            self._wake.notify()
            return record

    def pause(self, job_id: str) -> JobRecord | None:
        """Cooperatively pause a running job at the next pipeline checkpoint."""
        with self._wake:
            record = self._jobs.get(job_id)
            if record is None or record.status in {"completed", "failed", "cancelled"}:
                return record
            record.pause_event.set()
            if record.status in {"queued", "validating", "running"}:
                record.status = "paused"
                record.current_stage = "DuraklatÄ±ldÄ±"
                record.message = "Analiz duraklatÄ±ldÄ±; devam etmek iÃ§in Resume kullanÄ±n."
            self._wake.notify_all()
            return record

    def resume(self, job_id: str) -> JobRecord | None:
        """Release a paused job; the worker continues from its next checkpoint."""
        with self._wake:
            record = self._jobs.get(job_id)
            if record is None or record.status in {"completed", "failed", "cancelled"}:
                return record
            record.pause_event.clear()
            if record.status == "paused":
                record.status = "running"
                record.current_stage = "Analiz"
                record.message = "Analiz devam ediyor."
            self._wake.notify_all()
            return record

    def _checkpoint(self, record: JobRecord) -> None:
        """Wait without holding the manager lock so pause/resume/cancel remain responsive."""
        with self._wake:
            while record.pause_event.is_set() and not record.cancel_event.is_set():
                record.status = "paused"
                record.current_stage = "DuraklatÄ±ldÄ±"
                self._wake.wait(timeout=0.25)
            if record.cancel_event.is_set():
                from ..pipeline import AnalysisCancelled
                raise AnalysisCancelled()

    def _worker_loop(self) -> None:
        while True:
            with self._wake:
                while not self._queue:
                    self._wake.wait()
                job_id = self._queue.popleft()
                record = self._jobs.get(job_id)
                if record is None or record.status == "cancelled":
                    continue
            self._run(record)

    def _run(self, record: JobRecord) -> None:
        try:
            self._checkpoint(record)
            self._set(record, status="validating", current_stage="Doğrulanıyor", message="Video, modeller ve ayarlar doğrulanıyor.", started_at=_now())
            source = validate_video_path(record.source_path)
            from ..cli import _default_pose_path
            from ..config import load_config
            from ..model_loader import resolve_model_file_paths
            from ..video_io import read_metadata

            options = record.request.options
            metadata = read_metadata(source)
            self._checkpoint(record)
            health_model, instrument_model = resolve_model_file_paths(self.settings.project_root, None, None, auto_download=False)
            pose_model = _default_pose_path(self.settings.project_root)
            if options.privacy_mode in {"skeleton-only", "both"} and not pose_model.is_file():
                raise ValueError("Skeleton analizi için doğrulanmış pose modeli bulunamadı.")
            if record.cancel_event.is_set():
                self._cancelled(record)
                return
            overrides: dict = {
                "runtime": {"output_dir": str(record.job_root), "device": options.device},
                "depth": {"enabled": options.depth_enabled},
                "tracking": {"algorithm": options.tracker},
                "cli": {"privacy_mode": options.privacy_mode},
            }
            if options.health_confidence is not None:
                overrides["tracking"]["health"] = {"confidence": options.health_confidence}
            if options.instrument_confidence is not None:
                overrides["tracking"]["instrument"] = {"confidence": options.instrument_confidence}
            if options.iou is not None:
                overrides["tracking"]["health"] = {**overrides["tracking"].get("health", {}), "iou": options.iou}
                overrides["tracking"]["instrument"] = {**overrides["tracking"].get("instrument", {}), "iou": options.iou}
            config = load_config(self.settings.project_root, overrides)
            self._set(record, total_frames=metadata.frame_count)
            from ..pipeline import AnalysisCancelled, AnalysisPipeline
            from ..pose_pipeline import run_pose_pilot

            def progress(start: float, end: float):
                def update(processed: int, total: int, eta: float, stage: str) -> None:
                    self._checkpoint(record)
                    scaled = start + (end - start) * (processed / total if total else 0.0)
                    self._set(
                        record,
                        status="running",
                        progress_percent=min(99.0, scaled),
                        processed_frames=processed,
                        total_frames=total,
                        estimated_remaining_seconds=max(0.0, eta),
                        current_stage=stage,
                        message="Analiz sürüyor.",
                    )
                return update

            self._set(record, status="running", current_stage="Analiz", message="Analiz motoru başlatıldı.")
            if options.render_mode != "legacy":
                from ..unified import SelectionEvent, UnifiedAnalysisPipeline

                events = [SelectionEvent.from_dict(item) for item in options.selection_events]
                UnifiedAnalysisPipeline(self.settings.project_root, config).run(
                    source, record.job_root, health_model, instrument_model, pose_model,
                    render_mode=options.render_mode, enable_sam3=options.enable_sam3,
                    enable_xray_skeleton=options.enable_xray_skeleton, selection_events=events,
                    progress=progress(0, 99), cancel_event=record.cancel_event,
                )
            elif options.privacy_mode == "blur":
                AnalysisPipeline(self.settings.project_root, config).run(source, health_model, instrument_model, progress(0, 99), record.cancel_event)
            elif options.privacy_mode == "skeleton-only":
                run_pose_pilot(source, health_model, instrument_model, record.job_root, pose_model, config, keypoint_confidence=options.pose_confidence, progress=progress(0, 99), cancel_event=record.cancel_event)
            else:
                AnalysisPipeline(self.settings.project_root, config).run(source, health_model, instrument_model, progress(0, 50), record.cancel_event)
                run_pose_pilot(source, health_model, instrument_model, record.job_root, pose_model, config, keypoint_confidence=options.pose_confidence, progress=progress(50, 99), cancel_event=record.cancel_event)
            self._set(record, status="finalizing", progress_percent=99.0, current_stage="Sonuçlar hazırlanıyor", message="Artifact dosyaları güvenli olarak listeleniyor.")
            record.artifacts = collect_artifacts(record.job_root)
            self._set(record, status="completed", progress_percent=100.0, current_stage="Tamamlandı", message="Analiz tamamlandı.", finished_at=_now(), estimated_remaining_seconds=0.0)
        except Exception as error:
            from ..pipeline import AnalysisCancelled

            if isinstance(error, AnalysisCancelled) or record.cancel_event.is_set():
                self._cancelled(record)
            else:
                self._set(
                    record,
                    status="failed",
                    current_stage="Başarısız",
                    message="Analiz tamamlanamadı.",
                    error_code="ANALYSIS_FAILED",
                    safe_error_message="İşlem tamamlanamadı. Model, video ve sistem durumunu Doctor ekranından kontrol edin.",
                    finished_at=_now(),
                )
        finally:
            self._cleanup_temporary_upload(record)

    def _cancelled(self, record: JobRecord) -> None:
        self._set(
            record,
            status="cancelled",
            current_stage="İptal edildi",
            message="Analiz güvenli bir sınırda iptal edildi; yarım çıktılar tamamlanmış sayılmaz.",
            finished_at=_now(),
            estimated_remaining_seconds=None,
        )

    def _set(self, record: JobRecord, **values: object) -> None:
        with self._lock:
            for key, value in values.items():
                setattr(record, key, value)

    def _cleanup_temporary_upload(self, record: JobRecord) -> None:
        """Remove a completed mobile upload only once no queued job still needs it."""
        try:
            uploads_root = self.settings.uploads_root.resolve()
            source = record.source_path.resolve()
            if source.parent != uploads_root:
                return
            with self._lock:
                still_needed = any(
                    other.job_id != record.job_id
                    and other.source_path == record.source_path
                    and other.status not in {"completed", "failed", "cancelled"}
                    for other in self._jobs.values()
                )
            if not still_needed:
                source.unlink(missing_ok=True)
        except OSError:
            # Cleanup is best effort and must never change an analysis result.
            return

    def artifacts(self, job_id: str) -> dict[str, Artifact] | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return record.artifacts if record else None
