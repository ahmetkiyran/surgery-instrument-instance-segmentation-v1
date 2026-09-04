[CmdletBinding()]
param(
    [int]$Port = 8765,
    [switch]$Lan,
    [string]$Token
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) { throw 'Eksik .venv\Scripts\python.exe. Önce proje-local Python ortamını kurun.' }
if ($Lan -and $Token.Length -lt 32) { throw 'LAN modu için en az 32 karakterlik -Token zorunludur.' }
Push-Location $Root
try {
    $Arguments = @('-m', 'surgical_pipeline.server', '--port', $Port, '--log-level', 'info')
    if ($Lan) { $Arguments += @('--host', '0.0.0.0', '--lan', '--token', $Token) }
    & $Python @Arguments
} finally { Pop-Location }
