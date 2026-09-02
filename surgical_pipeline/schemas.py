"""Data contracts shared by detection, analytics and output writers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


@dataclass(slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    track_id: int | None
    mask: np.ndarray
    frame_index: int
    timestamp_s: float

    def centroid(self) -> tuple[float, float] | None:
        points = np.argwhere(self.mask > 0)
        if points.size == 0:
            return None
        y, x = np.median(points, axis=0)
        return float(x), float(y)


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

    def row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class UsageInterval:
    class_name: str
    track_id: int
    start_s: float
    end_s: float
    duration_s: float

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
