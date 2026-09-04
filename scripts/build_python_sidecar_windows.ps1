[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) { throw 'Python sidecar için .venv\Scripts\python.exe bulunamadı.' }
& $Python -m PyInstaller --version *> $null
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller bu proje ortamında kurulu değil. .venv içinde pip install pyinstaller çalıştırın.' }
Push-Location $Root
try {
    & $Python -m PyInstaller --noconfirm --clean --name surgical-api --onefile --collect-all surgical_pipeline --collect-all fastapi --collect-all uvicorn surgical_pipeline\server\__main__.py
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller sidecar derlemesi başarısız oldu.' }
    $Target = Join-Path $Root 'apps\desktop\src-tauri\binaries'
    New-Item -ItemType Directory -Force -Path $Target | Out-Null
    Copy-Item (Join-Path $Root 'dist\surgical-api.exe') (Join-Path $Target 'surgical-api-x86_64-pc-windows-msvc.exe') -Force
    Write-Host 'Sidecar hazır. Model ağırlıkları binary içine alınmadı; doğrulanmış models/weights dizininden çözülür.' -ForegroundColor Green
} finally { Pop-Location }
