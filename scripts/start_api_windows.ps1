[CmdletBinding()]
param(
    [int]$Port = 8765,
    [switch]$Lan,
    [string]$Token,
    [string]$PythonPath
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$ProjectPython = Join-Path $Root '.venv\Scripts\python.exe'
$SiblingPython = Join-Path (Split-Path -Parent $Root) 'sam3tracking\.venv\Scripts\python.exe'
$Python = if ($PythonPath) { $PythonPath } elseif ($env:SAM3_PYTHON_PATH) { $env:SAM3_PYTHON_PATH } elseif (Test-Path $SiblingPython) { $SiblingPython } else { $ProjectPython }
if (-not (Test-Path $Python)) { throw 'Eksik .venv\Scripts\python.exe. Önce proje-local Python ortamını kurun.' }
if ($Lan -and $Token.Length -lt 32) { throw 'LAN modu için en az 32 karakterlik -Token zorunludur.' }
Push-Location $Root
try {
    $Arguments = @('-m', 'surgical_pipeline.server', '--port', $Port, '--log-level', 'info')
    if ($Lan) { $Arguments += @('--host', '0.0.0.0', '--lan', '--token', $Token) }
    & $Python @Arguments
} finally { Pop-Location }
