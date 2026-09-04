[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw 'Node.js bulunamadı.' }
if (-not (Get-Command rustc -ErrorAction SilentlyContinue)) { throw 'Rust toolchain bulunamadı; Tauri için rustup ile Rust yükleyin.' }
if (-not (Test-Path (Join-Path $Root 'node_modules'))) { throw 'node_modules bulunamadı. Proje kökünde npm install çalıştırın.' }
Push-Location $Root
try { & npm.cmd run dev:desktop } finally { Pop-Location }
