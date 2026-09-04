"""`python -m surgical_pipeline.server` command entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .app import create_app
from .config import settings_from_environment


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Yerel cerrahi analiz API'si")
    command.add_argument("--host", help="Varsayılan: 127.0.0.1")
    command.add_argument("--port", type=int, help="Varsayılan: 8765")
    command.add_argument("--token", help="LAN veya yönetilen oturum için API tokenı")
    command.add_argument("--lan", action="store_true", help="Açıkça etkinleştirilen, token zorunlu yerel ağ modu")
    command.add_argument("--log-level", default="info", choices=("critical", "error", "warning", "info", "debug"))
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = Path.cwd().resolve()
    try:
        settings = settings_from_environment(root, host=args.host, port=args.port, token=args.token, lan_mode=args.lan or None)
    except ValueError as error:
        parser().error(str(error))
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level=args.log_level, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
