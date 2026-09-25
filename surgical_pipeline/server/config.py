"""Configuration for the deliberately local API server."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


@dataclass(frozen=True)
class ServerSettings:
    project_root: Path
    host: str = "127.0.0.1"
    port: int = 8765
    token: str | None = None
    lan_mode: bool = False
    max_upload_bytes: int = 20 * 1024 * 1024 * 1024
    max_upload_duration_seconds: float | None = 3600.0
    data_root_override: Path | None = None
    allowed_origins: tuple[str, ...] = (
        "http://localhost:1420",
        "http://tauri.localhost",
        "tauri://localhost",
    )

    @property
    def data_root(self) -> Path:
        return (self.data_root_override or (self.project_root / ".surgical_server")).resolve()

    @property
    def uploads_root(self) -> Path:
        return self.data_root / "uploads"

    @property
    def jobs_root(self) -> Path:
        return self.data_root / "jobs"

    @property
    def requires_token(self) -> bool:
        return self.lan_mode or self.host not in LOOPBACK_HOSTS

    def validate(self) -> "ServerSettings":
        if not 1 <= self.port <= 65535:
            raise ValueError("Port 1 ile 65535 arasında olmalıdır.")
        if self.host not in LOOPBACK_HOSTS and not self.lan_mode:
            raise ValueError("Loopback dışı bir host yalnızca --lan ile kullanılabilir.")
        if self.requires_token and (self.token is None or len(self.token) < 32):
            raise ValueError("LAN modu için en az 32 karakterlik SURGICAL_API_TOKEN gereklidir.")
        if self.max_upload_bytes <= 0:
            raise ValueError("Maksimum upload boyutu pozitif olmalıdır.")
        if self.max_upload_duration_seconds is not None and self.max_upload_duration_seconds <= 0:
            raise ValueError("Maksimum upload süresi pozitif olmalıdır.")
        return self


def settings_from_environment(project_root: Path, **overrides: object) -> ServerSettings:
    """Load only local-server values; tokens are never logged or returned."""
    raw_origins = os.environ.get("SURGICAL_API_ALLOWED_ORIGINS", "")
    origins = tuple(item.strip() for item in raw_origins.split(",") if item.strip())
    env_lan = os.environ.get("SURGICAL_API_LAN_MODE", "").strip().casefold() in {"1", "true", "yes"}
    values: dict[str, object] = {
        "project_root": project_root.resolve(),
        "host": os.environ.get("SURGICAL_API_HOST", "127.0.0.1"),
        "port": int(os.environ.get("SURGICAL_API_PORT", "8765")),
        "token": os.environ.get("SURGICAL_API_TOKEN") or None,
        "lan_mode": env_lan,
        "max_upload_bytes": int(os.environ.get("SURGICAL_API_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024 * 1024))),
        "max_upload_duration_seconds": float(os.environ.get("SURGICAL_API_MAX_UPLOAD_SECONDS", "3600")),
        "data_root_override": Path(os.environ["SURGICAL_API_DATA_ROOT"]).expanduser() if os.environ.get("SURGICAL_API_DATA_ROOT") else None,
        "allowed_origins": origins or ServerSettings.allowed_origins,
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    return ServerSettings(**values).validate()
