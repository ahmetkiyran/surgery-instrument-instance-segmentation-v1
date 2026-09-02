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
    if (-not (Test-Path (Join-Path $Root '.env'))) {
        & $VenvPython -m surgical_pipeline models download
        if ($LASTEXITCODE -ne 0) { throw 'Release modelleri indirilemedi. README içindeki manuel/offline kurulumu izleyin.' }
    }
    & $VenvPython -m surgical_pipeline doctor
    if ($LASTEXITCODE -ne 0) { throw 'Ön kontrol başarısız. Yukarıdaki [FAIL] satırlarını düzeltin.' }
    & $VenvPython app.py
} finally {
    Pop-Location
}
