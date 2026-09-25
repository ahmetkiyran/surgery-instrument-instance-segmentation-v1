[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $VenvPython)) {
    Write-Host 'Sanal ortam bulunamadı; ilk kurulum başlatılıyor...' -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot 'setup_windows.ps1')
}
Push-Location $Root
try {
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { throw 'FFmpeg/FFprobe PATH üzerinde bulunamadı; önce FFmpeg kurun.' }
    & $VenvPython -m surgical_pipeline models verify
    if ($LASTEXITCODE -ne 0) { throw 'Modeller eksik veya doğrulanamadı. README içindeki model kurulum talimatlarını izleyin.' }
    & $VenvPython -m surgical_pipeline doctor
    if ($LASTEXITCODE -ne 0) { throw 'Ön kontrol başarısız. Yukarıdaki [FAIL] satırlarını düzeltin.' }
    & $VenvPython app.py
} finally {
    Pop-Location
}
