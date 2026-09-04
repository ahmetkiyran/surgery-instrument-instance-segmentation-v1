[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw 'Node.js bulunamadı.' }
if (-not (Test-Path (Join-Path $Root 'node_modules'))) { throw 'node_modules bulunamadı. Proje kökünde npm install çalıştırın.' }
Write-Host 'Fiziksel cihaz için API’yi bilinçli LAN modunda başlatın:' -ForegroundColor Yellow
Write-Host '.\scripts\start_api_windows.ps1 -Lan -Token <en-az-32-karakter>' -ForegroundColor Yellow
Write-Host 'Uygulamada bilgisayarın LAN IP adresini ve aynı tokenı girin. Port yönlendirme veya public erişim kullanmayın.' -ForegroundColor Yellow
Push-Location $Root
try { & npm.cmd run start:mobile } finally { Pop-Location }
