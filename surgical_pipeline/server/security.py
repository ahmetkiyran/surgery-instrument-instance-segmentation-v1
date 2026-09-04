"""Authentication and path guards for the local service boundary."""

from __future__ import annotations

import hmac
import re
from pathlib import Path

from fastapi import HTTPException, Request, status

from .config import LOOPBACK_HOSTS, ServerSettings


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def is_loopback_request(request: Request) -> bool:
    return bool(request.client and request.client.host in LOOPBACK_HOSTS)


def extract_token(request: Request) -> str | None:
    bearer = request.headers.get("authorization", "")
    if bearer.lower().startswith("bearer "):
        return bearer[7:].strip()
    return request.headers.get("x-api-token")


async def require_api_access(request: Request) -> None:
    settings: ServerSettings = request.app.state.settings
    supplied = extract_token(request)
    if settings.token and supplied and hmac.compare_digest(supplied, settings.token):
        return
    if is_loopback_request(request) and not settings.requires_token and not settings.token:
        return
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Kimlik doğrulaması gerekli.")


def require_loopback(request: Request) -> None:
    if not is_loopback_request(request):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Bu işlem yalnızca yerel masaüstü bağlantısından yapılabilir.")


def sanitize_filename(value: str | None) -> str:
    filename = Path(value or "video").name
    sanitized = _UNSAFE_NAME.sub("_", filename).strip("._")[:120]
    return sanitized or "video"


def validate_video_path(path: Path) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise ValueError("Desteklenen, okunabilir bir video dosyasÄ± seÃ§in.") from error
    if not resolved.is_file() or resolved.suffix.casefold() not in VIDEO_SUFFIXES:
        raise ValueError("Desteklenen, okunabilir bir video dosyası seçin.")
    return resolved


def ensure_child(path: Path, root: Path) -> Path:
    candidate = path.resolve(strict=False)
    parent = root.resolve(strict=False)
    if candidate != parent and parent not in candidate.parents:
        raise ValueError("Güvenli olmayan dosya yolu.")
    return candidate
