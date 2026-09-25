#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  "$ROOT/scripts/setup_linux.sh"
fi
cd "$ROOT"
command -v ffmpeg >/dev/null 2>&1 || { echo "FFmpeg/FFprobe PATH üzerinde bulunamadı; önce FFmpeg kurun." >&2; exit 1; }
"$ROOT/.venv/bin/python" -m surgical_pipeline models verify
"$ROOT/.venv/bin/python" -m surgical_pipeline doctor
"$ROOT/.venv/bin/python" app.py
