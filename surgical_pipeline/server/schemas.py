"""Public Pydantic contracts for the local API. They intentionally omit local paths."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


JobStatus = Literal["queued", "validating", "running", "finalizing", "completed", "failed", "cancelled"]
PrivacyMode = Literal["skeleton-only", "blur", "both"]


class ArtifactResponse(BaseModel):
    artifact_id: str
    filename: str
    kind: str
    size_bytes: int
    media_type: str
    sensitive: bool = False


class JobOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    privacy_mode: PrivacyMode = "skeleton-only"
    device: str = "auto"
    depth_enabled: bool = True
    health_confidence: float | None = Field(default=None, gt=0, le=1)
    instrument_confidence: float | None = Field(default=None, gt=0, le=1)
    pose_confidence: float = Field(default=0.25, gt=0, le=1)
    iou: float | None = Field(default=None, gt=0, le=1)
    tracker: Literal["botsort", "bytetrack"] = "botsort"
    output_directory: str | None = Field(default=None, max_length=4096)

    @field_validator("device")
    @classmethod
    def device_is_safe(cls, value: str) -> str:
        normalized = value.strip().casefold()
        if normalized in {"auto", "cpu"} or normalized.isdecimal() or re.fullmatch(r"cuda(?::\d+)?", normalized):
            return value.strip()
        raise ValueError("device auto, cpu, cuda[:N] veya GPU indeksi olmalıdır.")


class JobCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upload_id: str | None = Field(default=None, pattern=r"^[a-f0-9-]{36}$")
    local_path: str | None = Field(default=None, max_length=4096)
    options: JobOptions = Field(default_factory=JobOptions)

    @field_validator("local_path")
    @classmethod
    def local_path_not_blank(cls, value: str | None) -> str | None:
        return value.strip() if value and value.strip() else None


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress_percent: float = Field(ge=0, le=100)
    processed_frames: int = Field(ge=0)
    total_frames: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    estimated_remaining_seconds: float | None = Field(default=None, ge=0)
    current_stage: str
    message: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    privacy_mode: PrivacyMode
    artifacts: list[ArtifactResponse] = Field(default_factory=list)
    error_code: str | None = None
    safe_error_message: str | None = None


class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    size_bytes: int
    duration_seconds: float


class LocalVideoInspectionRequest(BaseModel):
    """Desktop-only local-path request; the path is never echoed in a response."""

    model_config = ConfigDict(extra="forbid")

    local_path: str = Field(min_length=1, max_length=4096)

    @field_validator("local_path")
    @classmethod
    def local_path_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Yerel video yolu boÅŸ olamaz.")
        return value


class LocalVideoInspectionResponse(BaseModel):
    filename: str
    size_bytes: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    api_version: str = "v1"
    auth_required: bool
    lan_mode: bool


class DoctorCheckResponse(BaseModel):
    label: str
    ok: bool
    detail: str
    severity: Literal["critical", "warning"]


class DoctorResponse(BaseModel):
    ready: bool
    checks: list[DoctorCheckResponse]


class ModelResponse(BaseModel):
    key: str
    state: str
    detail: str
    ready: bool


class DefaultsResponse(BaseModel):
    privacy_mode: PrivacyMode
    device: str
    depth_enabled: bool
    health_confidence: float
    instrument_confidence: float
    iou: float
    tracker: Literal["botsort", "bytetrack"]
