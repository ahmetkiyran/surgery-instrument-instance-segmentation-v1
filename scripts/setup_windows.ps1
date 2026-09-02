[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root '.venv'
if (-not (Test-Path (Join-Path $Venv 'Scripts\python.exe'))) {
    $Python = Get-Command py -ErrorAction SilentlyContinue
    if ($Python) {
        & py -3.11 -m venv $Venv
    } else {
        & python -m venv $Venv
    }
}
$Py = Join-Path $Venv 'Scripts\python.exe'
& $Py -m pip install --upgrade pip
& $Py -m pip install -r (Join-Path $Root 'requirements-dev.txt')
if (-not (Test-Path (Join-Path $Root '.env'))) {
    Push-Location $Root
    try {
        & $Py -m surgical_pipeline models download
        if ($LASTEXITCODE -ne 0) {
            Write-Host 'Release modelleri indirilemedi. README içindeki manuel/offline kurulumu izleyin.' -ForegroundColor Yellow
        }
    } finally {
        Pop-Location
    }
}
Write-Host "Kurulum tamamlandı: $Venv" -ForegroundColor Cyan
Write-Host "Model yolları için .env dosyasını kontrol edin; .env.example güvenli şablondur." -ForegroundColor Yellow
