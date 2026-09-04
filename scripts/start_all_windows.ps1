[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) { throw 'Eksik .venv\Scripts\python.exe. Önce proje-local Python ortamını kurun.' }
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw 'Node.js bulunamadı.' }
if (-not (Get-Command rustc -ErrorAction SilentlyContinue)) { throw 'Rust toolchain bulunamadı; Tauri başlatılamaz.' }
if (-not (Test-Path (Join-Path $Root 'node_modules'))) { throw 'node_modules bulunamadı. Proje kökünde npm install çalıştırın.' }

$Api = Start-Process -FilePath $Python -ArgumentList '-m', 'surgical_pipeline.server', '--host', '127.0.0.1', '--port', '8765' -WorkingDirectory $Root -WindowStyle Hidden -PassThru
try {
    $Ready = $false
    for ($Attempt = 0; $Attempt -lt 20; $Attempt++) {
        try {
            $Response = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8765/api/v1/health' -TimeoutSec 2
            if ($Response.StatusCode -eq 200) { $Ready = $true; break }
        } catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $Ready) { throw 'Yerel API 10 saniye içinde hazır olmadı.' }
    Push-Location $Root
    try { & npm.cmd run dev:desktop } finally { Pop-Location }
} finally {
    if (-not $Api.HasExited) { Stop-Process -Id $Api.Id -Force }
}
