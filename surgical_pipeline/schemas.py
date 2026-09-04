"""Data contracts shared by detection, analytics and output writers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


PIPELINE_SCHEMA_VERSION = "1.0"


@dataclass(slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    track_id: int | None
    mask: np.ndarray
    frame_index: int
    timestamp_s: float
    bbox_xyxy: tuple[float, float, float, float] | None = None
    mask_quality: str = "valid"
    tracking_state: str = "unassigned"

    def centroid(self) -> tuple[float, float] | None:
        points = np.argwhere(self.mask > 0)
        if points.size:
            y, x = np.median(points, axis=0)
            return float(x), float(y)
        if self.bbox_xyxy is not None:
            x1, y1, x2, y2 = self.bbox_xyxy
            return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)
        return None

    def centroid_source(self) -> str:
        return "mask_centroid" if np.any(self.mask > 0) else "bbox_center" if self.bbox_xyxy is not None else "unavailable"


@dataclass(slots=True)
class TrackPoint:
    frame_index: int
    timestamp_s: float
    track_id: int
    class_name: str
    confidence: float
    x: float | None
    y: float | None
    depth: float | None
    relative_x: float | None
    relative_y: float | None
    relative_depth: float | None
    depth_valid: bool
    centroid_source: str = "mask_centroid"
    depth_source: str = "mask_median"
    depth_interpolated: bool = False

    def row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class UsageInterval:
    class_name: str
    track_id: int
    start_s: float
    end_s: float
    duration_s: float
    source: str = "track"

    def row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelInfo:
    path: str
    sha256: str
    names: dict[int, str]
    task: str

    def safe_dict(self) -> dict[str, Any]:
        # Paths intentionally excluded from artifacts destined for public sharing.
        return {"sha256": self.sha256, "classes": self.names, "task": self.task}


@dataclass
class AnalysisState:
    health_counts: list[dict[str, float | int]] = field(default_factory=list)
    track_points: list[TrackPoint] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SummarySchema(BaseModel):
    """Minimal, versioned contract for a completed analysis summary."""

    model_config = ConfigDict(extra="allow")

    schema_version: str
    video: dict[str, Any]
    models: dict[str, Any]
    health_person_statistics: dict[str, Any]
    instrument_usage: dict[str, Any]
    relative_3d_note: str
    processing: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)


class RunManifestSchema(BaseModel):
    """Strict envelope for a run lifecycle record; details may evolve independently."""

    model_config = ConfigDict(extra="allow")

    schema_version: str
    pipeline_version: str
    run_id: str
    status: str
    started_at_utc: str
    finished_at_utc: str | None = None
    warnings: list[str] = Field(default_factory=list)


class QualityReportSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str
    privacy: dict[str, Any]
    depth: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
