[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root '.venv'
if (-not (Test-Path (Join-Path $Venv 'Scripts\python.exe'))) {
    $Python = Get-Command py -ErrorAction SilentlyContinue
    if ($Python) { & py -3.11 --version *> $null }
    if ($Python -and $LASTEXITCODE -eq 0) {
        & py -3.11 -m venv $Venv
    } else {
        & python -m venv $Venv
    }
}
$Py = Join-Path $Venv 'Scripts\python.exe'
$Version = & $Py -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
if ($Version -notin @('3.11', '3.12')) { throw "Python 3.11 veya 3.12 gerekir; bulunan sürüm: $Version" }
& $Py -m pip install --upgrade pip
& $Py -m pip install -r (Join-Path $Root 'requirements-dev.txt')
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host 'FFmpeg bulunamadı; video analizi başlamadan önce FFmpeg/FFprobe PATH içine eklenmelidir.' -ForegroundColor Yellow
}
Push-Location $Root
try {
    & $Py -m surgical_pipeline models verify
    if ($LASTEXITCODE -ne 0) { Write-Host 'Modeller eksik veya doğrulanamadı; gerçek model dosyalarını README talimatıyla ekleyin.' -ForegroundColor Yellow }
} finally { Pop-Location }
Write-Host "Kurulum tamamlandı: $Venv" -ForegroundColor Cyan
Write-Host "SAM3 isteğe bağlıdır; temel v1 akışı için .env.example yalnızca gerektiğinde kopyalanır." -ForegroundColor Yellow
