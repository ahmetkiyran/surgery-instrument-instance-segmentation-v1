[CmdletBinding()]
param(
  [switch]$EnableDepth,
  [int]$UnitBuildTimeoutMinutes = 20,
  [int]$PythonCoreTimeoutMinutes = 30,
  [int]$CliTimeoutMinutes = 30,
  [int]$DesktopTimeoutMinutes = 40,
  [int]$MobileTimeoutMinutes = 45
)

$ErrorActionPreference = "Continue"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RunId = Get-Date -Format "yyyyMMdd_HHmmss"
$ReportRoot = Join-Path $ProjectRoot "test_reports\full_acceptance_$RunId"
$Logs = Join-Path $ReportRoot "logs"
$E2e = Join-Path $ProjectRoot "outputs\e2e\$RunId"
$Temp = Join-Path $ProjectRoot "tmp\e2e"
$Clip = Join-Path $Temp "kaynak_video_10s.mp4"
$Stages = New-Object System.Collections.Generic.List[object]
New-Item -ItemType Directory -Force -Path $Logs, (Join-Path $ReportRoot "screenshots"), $E2e, $Temp | Out-Null

function Q([string]$Value) {
  if ($Value -notmatch '[\s"]') { return $Value }
  '"' + ($Value -replace '"', '\"') + '"'
}
function Record([string]$Name,[string]$Status,[int]$Code,[datetime]$Start,[string]$Log,[string]$Detail) {
  $End=Get-Date
  $Stages.Add([pscustomobject]@{name=$Name;status=$Status;exit_code=$Code;started_at=$Start.ToUniversalTime().ToString("o");finished_at=$End.ToUniversalTime().ToString("o");elapsed_seconds=[math]::Round(($End-$Start).TotalSeconds,3);log=$Log;detail=$Detail})
}
function Run([string]$Name,[string]$File,[string[]]$Arguments,[string]$Log,[int]$Minutes=20,[string]$Cwd=$ProjectRoot) {
  $Start=Get-Date; $Err="$Log.stderr"; $Line=[string]::Join(" ",@($Arguments|ForEach-Object{Q ([string]$_)}))
  "Started: $($Start.ToUniversalTime().ToString('o'))" | Set-Content $Log -Encoding utf8
  try {
    # Start-Process does not reliably populate ExitCode after WaitForExit on every
    # supported PowerShell/.NET combination.  Use Process directly so a failed
    # subprocess can never be reported as a passing acceptance stage.
    $Info=New-Object System.Diagnostics.ProcessStartInfo
    $ResolvedCommand=Get-Command $File -CommandType Application -ErrorAction Stop | Select-Object -First 1
    $Info.FileName=if($ResolvedCommand.Source){$ResolvedCommand.Source}else{$ResolvedCommand.Path}; $Info.Arguments=$Line; $Info.WorkingDirectory=$Cwd
    $Info.UseShellExecute=$false; $Info.CreateNoWindow=$true
    $Info.RedirectStandardOutput=$true; $Info.RedirectStandardError=$true
    $P=New-Object System.Diagnostics.Process; $P.StartInfo=$Info
    if(-not $P.Start()){throw "Could not start: $File"}
    $OutputTask=$P.StandardOutput.ReadToEndAsync(); $ErrorTask=$P.StandardError.ReadToEndAsync()
    if(-not $P.WaitForExit([math]::Max(1,$Minutes)*60000)){
      Stop-Process -Id $P.Id -Force -ErrorAction SilentlyContinue; $P.WaitForExit()
      $OutputTask.Result|Add-Content $Log; $ErrorTask.Result|Set-Content $Err -Encoding utf8
      if(Test-Path $Err){Get-Content $Err|Add-Content $Log}; Record $Name "FAILED" 124 $Start $Log "Timeout; only the process started by this runner was stopped."
      return [pscustomobject]@{status="FAILED";exit_code=124}
    }
    $OutputTask.Result|Add-Content $Log; $ErrorTask.Result|Set-Content $Err -Encoding utf8
    $ExitCode = [int]$P.ExitCode; $S = "FAILED"; if($ExitCode -eq 0){$S = "PASSED"}
    if(Test-Path $Err){Get-Content $Err|Add-Content $Log}
    Record $Name $S $ExitCode $Start $Log "Command completed."; return [pscustomobject]@{status=$S;exit_code=$ExitCode}
  } catch {
    $_|Out-String|Add-Content $Log; Record $Name "BLOCKED" 127 $Start $Log $_.Exception.Message
    return [pscustomobject]@{status="BLOCKED";exit_code=127}
  }
}
function Mark([string]$Name,[string]$Status,[string]$Detail) {
  $Log=Join-Path $Logs "$Name.log"; $Detail|Set-Content $Log -Encoding utf8; Record $Name $Status 0 (Get-Date) $Log $Detail
  [pscustomobject]@{status=$Status;exit_code=0}
}
function Snapshot([string]$Path) {
  $I=Get-Item -LiteralPath $Path
  $P=& ffprobe -v error -show_entries format=duration,format_name:stream=codec_type,codec_name,width,height,avg_frame_rate -of json -- $Path|ConvertFrom-Json
  $V=@($P.streams|Where-Object{$_.codec_type-eq"video"})|Select-Object -First 1; $A=@($P.streams|Where-Object{$_.codec_type-eq"audio"})
  [pscustomobject]@{filename=$I.Name;size_bytes=$I.Length;modified_utc=$I.LastWriteTimeUtc.ToString("o");sha256=((Get-FileHash $Path -Algorithm SHA256).Hash.ToLowerInvariant());format=$P.format.format_name;duration_seconds=[double]$P.format.duration;fps=if($V){$V.avg_frame_rate}else{$null};width=if($V){$V.width}else{$null};height=if($V){$V.height}else{$null};video_codec=if($V){$V.codec_name}else{$null};audio_stream_count=@($A).Count;audio_codecs=@($A|ForEach-Object{$_.codec_name})}
}

Set-Location $ProjectRoot
$Python=if(Test-Path (Join-Path $ProjectRoot ".venv\Scripts\python.exe")){Join-Path $ProjectRoot ".venv\Scripts\python.exe"}else{"python"}
$Folders=@([Environment]::GetFolderPath("Desktop"));if($env:OneDrive){$Folders+=Join-Path $env:OneDrive "Desktop"};if($env:OneDriveCommercial){$Folders+=Join-Path $env:OneDriveCommercial "Desktop"}
$Rank=@{".mp4"=0;".mov"=1;".mkv"=2;".avi"=3;".m4v"=4}
$Candidates=foreach($Folder in ($Folders|Select-Object -Unique)){if(Test-Path $Folder){Get-ChildItem $Folder -File -ErrorAction SilentlyContinue|Where-Object{$_.BaseName -ieq "kaynak video" -and $Rank.ContainsKey($_.Extension.ToLowerInvariant())}|ForEach-Object{[pscustomobject]@{full_path=$_.FullName;filename=$_.Name;extension=$_.Extension.ToLowerInvariant();extension_rank=$Rank[$_.Extension.ToLowerInvariant()];size_bytes=$_.Length;modified_utc=$_.LastWriteTimeUtc.ToString("o")}}}}
$Candidates=@($Candidates|Sort-Object extension_rank,@{Expression={[datetime]$_.modified_utc};Descending=$true},full_path)
$Candidates|ConvertTo-Json -Depth 5|Set-Content (Join-Path $ReportRoot "source_candidates.json") -Encoding utf8
$Source=$Candidates|Select-Object -First 1;$Before=$null;$ClipReady=$false
if(-not $Source){Mark "source_video_discovery" "BLOCKED" "No exact kaynak video candidate was found."|Out-Null}
elseif(-not(Get-Command ffmpeg -ErrorAction SilentlyContinue) -or -not(Get-Command ffprobe -ErrorAction SilentlyContinue)){Mark "source_video_discovery" "BLOCKED" "ffmpeg or ffprobe unavailable."|Out-Null}
else{
  $Before=Snapshot $Source.full_path
  @{selected=$Before;candidates=@($Candidates|Select-Object filename,extension,size_bytes,modified_utc,extension_rank)}|ConvertTo-Json -Depth 8|Set-Content (Join-Path $ReportRoot "source_video_check.json") -Encoding utf8
  @{full_path=$Source.full_path;before=$Before}|ConvertTo-Json -Depth 8|Set-Content (Join-Path $Logs "source_video_local.json") -Encoding utf8
  Mark "source_video_discovery" "PASSED" "Selected $($Source.filename) using extension priority then latest modification."|Out-Null
  $ClipStage=Run "create_10_second_clip" "ffmpeg" @("-y","-ss","0","-t","10","-i",$Source.full_path,"-map_metadata","-1","-map_chapters","-1","-an","-c:v","libx264","-preset","medium","-crf","18","-movflags","+faststart",$Clip) (Join-Path $Logs "clip_creation.log")
  if($ClipStage.status-eq"PASSED" -and(Test-Path $Clip)){$ClipInfo=Snapshot $Clip;$ClipReady=$ClipInfo.video_codec-eq"h264"-and$ClipInfo.audio_stream_count-eq 0-and$ClipInfo.duration_seconds-gt 0;@{status=if($ClipReady){"PASSED"}else{"FAILED"};clip=$ClipInfo}|ConvertTo-Json -Depth 8|Set-Content (Join-Path $ReportRoot "clip_check.json") -Encoding utf8}
}

$PyTests=Run "python_tests" $Python @("-m","pytest","-vv","--tb=long","--durations=30") (Join-Path $Logs "python_tests.log") $UnitBuildTimeoutMinutes
$CliHelp=Run "cli_help" $Python @("-m","surgical_pipeline","--help") (Join-Path $Logs "cli_help.log") $UnitBuildTimeoutMinutes
$CliVersion=Run "cli_version" $Python @("-m","surgical_pipeline","version") (Join-Path $Logs "cli_version.log") $UnitBuildTimeoutMinutes
$CliAnalyze=Run "cli_analyze_help" $Python @("-m","surgical_pipeline","analyze","--help") (Join-Path $Logs "cli_analyze_help.log") $UnitBuildTimeoutMinutes
$CliDoctor=Run "cli_doctor" $Python @("-m","surgical_pipeline","doctor") (Join-Path $Logs "cli_doctor.log") $UnitBuildTimeoutMinutes
if(Get-Command surgery-analyze -ErrorAction SilentlyContinue){Run "console_script_help" "surgery-analyze" @("--help") (Join-Path $Logs "console_script.log") $UnitBuildTimeoutMinutes|Out-Null}else{Mark "console_script" "SKIPPED" "Console entry point is not installed."|Out-Null}
$Inventory=Join-Path $ReportRoot "model_inventory.json";$Paths=Join-Path $Logs "model_paths.local.json"
$Models=Run "model_inventory" $Python @("scripts\acceptance_python_core.py","--project-root",$ProjectRoot,"--model-inventory",$Inventory,"--model-paths",$Paths) (Join-Path $Logs "model_inventory.log") $UnitBuildTimeoutMinutes
$ApiStatic=Run "api_static_checks" $Python @("scripts\acceptance_api_e2e.py","--project-root",$ProjectRoot,"--data-root",(Join-Path $E2e "api_static_server"),"--result",(Join-Path $ReportRoot "api_checks.json"),"--mode","static","--server-log",(Join-Path $Logs "api_static_server.log")) (Join-Path $Logs "api_tests.log") $UnitBuildTimeoutMinutes
$ApiTypes=Run "api_client_typecheck" "npm.cmd" @("run","typecheck:api-client") (Join-Path $Logs "api_client_typecheck.log") $UnitBuildTimeoutMinutes
$DesktopTypes=Run "desktop_typecheck" "npm.cmd" @("run","typecheck:desktop") (Join-Path $Logs "desktop_typecheck.log") $UnitBuildTimeoutMinutes
$DesktopBuild=Run "desktop_build" "npm.cmd" @("run","build:desktop") (Join-Path $Logs "desktop_build.log") $UnitBuildTimeoutMinutes
if(Get-Command cargo -ErrorAction SilentlyContinue){$Cargo=Run "cargo_check" "cargo" @("check","--manifest-path","apps\desktop\src-tauri\Cargo.toml") (Join-Path $Logs "cargo_check.log") $UnitBuildTimeoutMinutes}else{$Cargo=Mark "cargo_check" "BLOCKED" "Rust/Cargo is unavailable; native Tauri check cannot run."}
$MobileTypes=Run "mobile_typecheck" "npm.cmd" @("run","typecheck:mobile") (Join-Path $Logs "mobile_build.log") $UnitBuildTimeoutMinutes
$Expo=Run "expo_doctor" "npx.cmd" @("expo-doctor") (Join-Path $Logs "expo_doctor.log") $UnitBuildTimeoutMinutes (Join-Path $ProjectRoot "apps\mobile")
$Android=@("adb","emulator","java")|ForEach-Object{$X=Get-Command $_ -ErrorAction SilentlyContinue;[pscustomobject]@{tool=$_;available=$null-ne$X;path=if($X){$X.Source}else{$null}}};$Android|ConvertTo-Json|Set-Content (Join-Path $ReportRoot "android_environment.json") -Encoding utf8

$CoreOut=Join-Path $E2e "python_core";$CliOut=Join-Path $E2e "cli";$DesktopOut=Join-Path $E2e "desktop";$MobileOut=Join-Path $E2e "mobile";$CoreJson=Join-Path $ReportRoot "python_core_e2e.json"
if($ClipReady-and$Models.status-eq"PASSED"){
  $CoreArgs=@("scripts\acceptance_python_core.py","--project-root",$ProjectRoot,"--model-inventory",$Inventory,"--model-paths",$Paths,"--input",$Clip,"--output",$CoreOut,"--result",$CoreJson);if($EnableDepth){$CoreArgs+="--enable-depth"}
  $Core=Run "python_core_e2e" $Python $CoreArgs (Join-Path $Logs "python_core_e2e.log") $PythonCoreTimeoutMinutes
  if($Core.status-eq"PASSED"){
    $M=Get-Content $Paths -Raw|ConvertFrom-Json
    $Cli=Run "cli_e2e" $Python @("-m","surgical_pipeline","analyze","--input",$Clip,"--health-model",$M.health,"--instrument-model",$M.instrument,"--pose-model",$M.pose,"--privacy-mode","skeleton-only","--no-depth","--output-dir",$CliOut) (Join-Path $Logs "cli_e2e.log") $CliTimeoutMinutes
    $Desktop=Run "desktop_api_contract_e2e" $Python @("scripts\acceptance_api_e2e.py","--project-root",$ProjectRoot,"--data-root",(Join-Path $DesktopOut "server_state"),"--result",(Join-Path $ReportRoot "desktop_e2e.json"),"--mode","desktop","--input",$Clip,"--output",$DesktopOut,"--server-log",(Join-Path $Logs "desktop_api_server.log")) (Join-Path $Logs "desktop_e2e.log") $DesktopTimeoutMinutes
    $Mobile=Run "mobile_api_contract_e2e" $Python @("scripts\acceptance_api_e2e.py","--project-root",$ProjectRoot,"--data-root",$MobileOut,"--result",(Join-Path $ReportRoot "mobile_e2e.json"),"--mode","mobile","--input",$Clip,"--server-log",(Join-Path $Logs "mobile_api_server.log")) (Join-Path $Logs "mobile_e2e.log") $MobileTimeoutMinutes
  }else{$Cli=Mark "cli_e2e" "BLOCKED" "Python core E2E failed.";$Desktop=Mark "desktop_api_contract_e2e" "BLOCKED" "Python core E2E failed.";$Mobile=Mark "mobile_api_contract_e2e" "BLOCKED" "Python core E2E failed."}
}else{$Core=Mark "python_core_e2e" "BLOCKED" "Clip or three-model inventory unavailable.";$Cli=Mark "cli_e2e" "BLOCKED" "Clip or three-model inventory unavailable.";$Desktop=Mark "desktop_api_contract_e2e" "BLOCKED" "Clip or three-model inventory unavailable.";$Mobile=Mark "mobile_api_contract_e2e" "BLOCKED" "Clip or three-model inventory unavailable."}

$Unchanged=$false;if($Before){$After=Snapshot $Source.full_path;$Unchanged=$Before.size_bytes-eq$After.size_bytes-and$Before.modified_utc-eq$After.modified_utc-and$Before.sha256-eq$After.sha256;@{selected=$After;unchanged=$Unchanged;candidates=@($Candidates|Select-Object filename,extension,size_bytes,modified_utc)}|ConvertTo-Json -Depth 8|Set-Content (Join-Path $ReportRoot "source_video_check.json") -Encoding utf8;Mark "source_integrity_after_tests" $(if($Unchanged){"PASSED"}else{"FAILED"}) "Compared size, timestamp and SHA-256."|Out-Null}
$CoreData=if(Test-Path $CoreJson){Get-Content $CoreJson -Raw|ConvertFrom-Json}else{$null};$DesktopData=if(Test-Path (Join-Path $ReportRoot "desktop_e2e.json")){Get-Content (Join-Path $ReportRoot "desktop_e2e.json") -Raw|ConvertFrom-Json}else{$null};$MobileData=if(Test-Path (Join-Path $ReportRoot "mobile_e2e.json")){Get-Content (Join-Path $ReportRoot "mobile_e2e.json") -Raw|ConvertFrom-Json}else{$null}
@{python_core=if($CoreData){$CoreData.artifacts}else{@()};cli=if(Test-Path $CliOut){@(Get-ChildItem $CliOut -File -Recurse|ForEach-Object{$_.Name})}else{@()};desktop_api_contract=if($DesktopData){$DesktopData.e2e.artifacts}else{@()};mobile_api_contract=if($MobileData){$MobileData.e2e.artifacts}else{@()}}|ConvertTo-Json -Depth 16|Set-Content (Join-Path $ReportRoot "artifact_manifest.json") -Encoding utf8
@{caveat="Skeleton outputs reduce exposure but do not guarantee complete anonymity.";python_core=if($CoreData){$CoreData.privacy_report}else{$null};desktop_api_contract=if($DesktopData){$DesktopData.e2e.privacy_report}else{$null};mobile_api_contract=if($MobileData){$MobileData.e2e.privacy_report}else{$null}}|ConvertTo-Json -Depth 20|Set-Content (Join-Path $ReportRoot "privacy_summary.json") -Encoding utf8
$DesktopContract=if($DesktopData){[string]$DesktopData.status}else{"BLOCKED"};$MobileContract=if($MobileData){[string]$MobileData.status}else{"BLOCKED"}
$Comparison=if($CoreData-and$Cli.status-eq"PASSED"-and$DesktopContract-eq"PASSED"-and$MobileContract-eq"PASSED"){"PARTIAL"}elseif($CoreData){"PARTIAL"}else{"BLOCKED"};@{status=$Comparison;desktop_api_contract=$DesktopContract;mobile_api_contract=$MobileContract;reason="Desktop and mobile runs are API-contract fallbacks, not native UI E2E."}|ConvertTo-Json|Set-Content (Join-Path $ReportRoot "comparison.json") -Encoding utf8
@{run_id=$RunId;python=(& $Python --version 2>&1|Out-String).Trim();node=(& node --version 2>&1|Out-String).Trim();npm=(& npm.cmd --version 2>&1|Out-String).Trim();cargo_available=[bool](Get-Command cargo -ErrorAction SilentlyContinue);android_tools=$Android;enable_depth=[bool]$EnableDepth}|ConvertTo-Json -Depth 8|Set-Content (Join-Path $ReportRoot "environment.json") -Encoding utf8
$Tauri=if($Cargo.status-eq"PASSED"-and$Desktop.status-eq"PASSED"){"PARTIAL"}else{"BLOCKED"};$AndroidReady=@($Android|Where-Object{-not $_.available}).Count-eq 0;$MobileNative=if($AndroidReady-and$Mobile.status-eq"PASSED"){"PARTIAL"}else{"BLOCKED"}
$Overall=if(-not$Unchanged-or$PyTests.status-eq"FAILED"-or$Core.status-eq"FAILED"-or$Cli.status-eq"FAILED"){"FAILED"}else{"PARTIAL"}
$Summary=@{overall=$Overall;source_found=$null-ne$Source;source_unchanged=$Unchanged;source_clip=if($ClipReady){Split-Path $Clip -Leaf}else{$null};python_tests=$PyTests.status;desktop_build=$DesktopBuild.status;mobile_build=$MobileTypes.status;python_core_e2e=$Core.status;cli_e2e=$Cli.status;desktop_api_contract_e2e=$DesktopContract;mobile_api_contract_e2e=$MobileContract;tauri_e2e=$Tauri;mobile_android_e2e=$MobileNative;privacy=if($CoreData){"CHECK_REPORT"}else{"BLOCKED"};comparison=$Comparison;report_root=$ReportRoot;output_directories=@{python_core=$CoreOut;cli=$CliOut;desktop=$DesktopOut;mobile=$MobileOut};stages=$Stages.ToArray()}
$Summary|ConvertTo-Json -Depth 24|Set-Content (Join-Path $ReportRoot "summary.json") -Encoding utf8
@"
# Full acceptance report - $RunId

Overall result: **$Overall**

| Environment | Unit/build | Real 10s E2E | Privacy | Result |
| --- | --- | --- | --- | --- |
| Python core | $($PyTests.status) | $($Core.status) | $(if($CoreData){'report generated'}else{'blocked'}) | $($Core.status) |
| CLI | $($CliDoctor.status) | $($Cli.status) | skeleton-only | $($Cli.status) |
| Tauri desktop | type=$($DesktopTypes.status), build=$($DesktopBuild.status), cargo=$($Cargo.status) | native UI $Tauri; API fallback $($Desktop.status) | $(if($DesktopData){'report generated'}else{'blocked'}) | $Tauri |
| Expo Android mobile | type=$($MobileTypes.status), doctor=$($Expo.status) | native Android $MobileNative; upload/API fallback $($Mobile.status) | $(if($MobileData){'report generated'}else{'blocked'}) | $MobileNative |

Source integrity: $(if($Unchanged){'PASSED'}else{'FAILED or unavailable'}). Native Tauri and Android tests are not claimed as passed without native tooling and automation.
"@|Set-Content (Join-Path $ReportRoot "summary.md") -Encoding utf8
git status --short|Set-Content (Join-Path $Logs "git_status.log") -Encoding utf8;git diff --check 2>&1|Set-Content (Join-Path $Logs "git_diff_check.log") -Encoding utf8;git ls-files|Set-Content (Join-Path $Logs "git_ls_files.log") -Encoding utf8
Write-Output "Acceptance report: $ReportRoot";Write-Output "Overall result: $Overall";exit $(if($Overall-eq"FAILED"){1}else{0})
