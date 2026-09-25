#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PY_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
case "$PY_VERSION" in
  3.11|3.12) ;;
  *) echo "Python 3.11 veya 3.12 gerekir; bulunan sürüm: $PY_VERSION" >&2; exit 1 ;;
esac
"$PYTHON_BIN" -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip
"$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements-dev.txt"
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "FFmpeg bulunamadı; video analizi başlamadan önce FFmpeg/FFprobe PATH içine eklenmelidir." >&2
fi
(cd "$ROOT" && "$ROOT/.venv/bin/python" -m surgical_pipeline models verify) || \
  echo "Modeller eksik veya doğrulanamadı; gerçek model dosyalarını README talimatıyla ekleyin." >&2
echo "Kurulum tamamlandı. SAM3 isteğe bağlıdır; .env.example yalnızca gerektiğinde kopyalanır."
