"""Session/selection state shared by web, desktop and Expo API clients."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
import time
from uuid import uuid4

from ..unified import SelectionEvent
from ..video_io import VideoMetadata, read_metadata


@dataclass
class AnalysisSession:
    session_id: str
    source: Path | None = None
    events: list[SelectionEvent] = field(default_factory=list)
    job_id: str | None = None
    paused: bool = False
    metadata: VideoMetadata | None = None
    current_frame: int = 0
    sam3_frames: dict[int, list[dict]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self, ttl_seconds: float = 24 * 60 * 60) -> None:
        self._items: dict[str, AnalysisSession] = {}
        self._lock = RLock()
        self.ttl_seconds = ttl_seconds

    def _cleanup_expired(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        expired = [key for key, value in self._items.items() if value.last_access < cutoff and value.job_id is None]
        for key in expired:
            self._items.pop(key, None)

    def create(self) -> AnalysisSession:
        with self._lock:
            self._cleanup_expired()
            value = AnalysisSession(str(uuid4()))
            self._items[value.session_id] = value
            return value

    def get(self, session_id: str) -> AnalysisSession | None:
        with self._lock:
            self._cleanup_expired()
            value = self._items.get(session_id)
            if value is not None:
                value.last_access = time.time()
            return value

    def add_event(self, session_id: str, event: SelectionEvent) -> AnalysisSession:
        with self._lock:
            item = self._items[session_id]
            if event.action == "remove" and event.unified_track_id:
                for existing in item.events:
                    if existing.unified_track_id == event.unified_track_id:
                        existing.action = "remove"
            item.events.append(event)
            return item

    def attach_source(self, session_id: str, source: Path) -> AnalysisSession:
        with self._lock:
            item = self._items[session_id]
            item.source, item.metadata, item.current_frame = source, read_metadata(source), 0
            return item

    def active_events(self, session_id: str) -> list[SelectionEvent]:
        with self._lock:
            item = self._items[session_id]
            removed = {event.event_id for event in item.events if event.action == "remove"}
            return [event for event in item.events if event.action in {"add", "select", "resume"} and event.event_id not in removed]
