#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip
"$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements-dev.txt"
if [[ ! -f "$ROOT/.env" ]]; then
  (cd "$ROOT" && "$ROOT/.venv/bin/python" -m surgical_pipeline models download) || \
    echo "Release modelleri indirilemedi; README içindeki manuel/offline kurulumu izleyin." >&2
fi
echo "Kurulum tamamlandı. .env.example dosyasını .env olarak kopyalayıp model yollarını ayarlayın."
