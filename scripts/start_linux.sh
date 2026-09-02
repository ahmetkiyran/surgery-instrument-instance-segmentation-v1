#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  "$ROOT/scripts/setup_linux.sh"
fi
cd "$ROOT"
if [[ ! -f "$ROOT/.env" ]]; then
  "$ROOT/.venv/bin/python" -m surgical_pipeline models download
fi
"$ROOT/.venv/bin/python" -m surgical_pipeline doctor
"$ROOT/.venv/bin/python" app.py
