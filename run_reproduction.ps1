<#
.SYNOPSIS
    Reproduce the best recorded main-pipeline arms on RootSE and TrajDebug SP.

.DESCRIPTION
    This script runs only the shared LongTraj main pipeline:
      flow tracing -> region prior -> compression/pruning -> localization.

    FlowMode=fresh rebuilds data/control flow from the raw trajectories.
    FlowMode=published copies the bundled flow artifacts and reuses only the
    graph, then reruns every downstream stage in a new output directory.

    API credentials are read from config.json; they are never embedded here.
#>

[CmdletBinding()]
param(
    [ValidateSet("all", "rootse", "sp")]
    [string]$Dataset = "all",

    [ValidateSet("fresh", "published")]
    [string]$FlowMode = "published",

    # highest: recorded 31/86, but that run had two failed API calls.
    # clean:   recorded 30/86 with zero failed API calls.
    [ValidateSet("highest", "clean")]
    [string]$SpArm = "highest",

    [string]$Model = "qwen3.7-plus",
    [string]$Python = "python",
    [string]$Config = (Join-Path $PSScriptRoot "config.json"),
    [string]$WorkspaceRoot = (Split-Path $PSScriptRoot -Parent),

    [ValidateRange(1, 32)]
    [int]$Workers = 4,

    [ValidateRange(1, 16)]
    [int]$SpanWorkers = 3,

    # 0 means the complete dataset. Use 1 for a paid end-to-end smoke run.
    [ValidateRange(0, 100000)]
    [int]$Limit = 0,

    # Exercise the whole orchestration with deterministic local fallbacks.
    # This checks plumbing only and does not produce a meaningful accuracy.
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$Repo = $PSScriptRoot
Set-Location -LiteralPath $Repo

function Require-Path {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Purpose
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing $Purpose`: $Path"
    }
}

if (-not $DryRun) {
    Require-Path $Config "API config"
}
Require-Path $WorkspaceRoot "workspace/data root"

$RunId = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir = Join-Path $Repo ("results/" + $RunId)
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    Write-Host ("$Python " + ($Arguments -join " ")) -ForegroundColor DarkGray
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

function Prepare-Trace {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$PublishedName
    )
    $destination = Join-Path $RunDir $Name
    if ($FlowMode -eq "published") {
        $source = Join-Path $Repo ("reference_flows/" + $PublishedName)
        Require-Path $source "published flow artifact"
        if (Test-Path -LiteralPath $destination) {
            throw "Trace destination already exists: $destination"
        }
        Copy-Item -LiteralPath $source -Destination $destination -Recurse
    }
    return $destination
}

function Add-CommonArgs {
    param([string[]]$BaseArgs)
    $out = @($BaseArgs) + @(
        "--root", $WorkspaceRoot,
        "--model", $Model,
        "--workers", "$Workers",
        "--span-workers", "$SpanWorkers",
        "--verbose"
    )
    if (Test-Path -LiteralPath $Config) {
        $out += @("--config", $Config)
    }
    if ($DryRun) {
        $out += "--dry-run"
    }
    if ($Limit -gt 0) {
        $out += @("--limit", "$Limit")
    }
    if ($FlowMode -eq "published") {
        $out += "--reuse-flow"
    }
    return $out
}

function Run-RootSE {
    Write-Host "`n=== RootSE / TrajAudit main pipeline ===" -ForegroundColor Cyan
    $trace = Prepare-Trace "trace_rootse" "rootse_q7long"
    $out = Join-Path $RunDir "rootse_q7long.jsonl"
    $commandArgs = Add-CommonArgs @(
        "-u", "-m", "longtraj.cli", "run",
        "--source", "trajaudit",
        "--span-prior",
        "--verbatim-limit", "2500",
        "--max-prompt-chars", "700000",
        "--trace-dir", $trace,
        "--out", $out
    )
    Invoke-Python @commandArgs
    Invoke-Python tools/split_eval.py $out
}

function Run-SP {
    Write-Host "`n=== TrajDebug SWE-bench Pro main pipeline ($SpArm) ===" -ForegroundColor Cyan
    if ($SpArm -eq "highest") {
        Write-Warning "The recorded 31/86 run had two failed calls; treat 0.3605 as a lower-bound reading."
        $published = "sp_fk0"
        $traceName = "trace_sp_highest"
        $outName = "sp_fk0.jsonl"
        $armArgs = @("--max-prompt-chars", "120000")
    } else {
        $published = "sp_capc1"
        $traceName = "trace_sp_clean"
        $outName = "sp_capc1.jsonl"
        $armArgs = @("--full-keep", "--max-prompt-chars", "700000")
    }
    $trace = Prepare-Trace $traceName $published
    $out = Join-Path $RunDir $outName
    $commandArgs = Add-CommonArgs (@(
        "-u", "-m", "longtraj.cli", "run",
        "--source", "trajdebug", "--subset", "swebenchpro",
        "--span-prior", "--unit-spans", "--prior-compress",
        "--candidates-strict",
        "--verify", "--verify-outdeg",
        "--no-outline", "--no-error-anchors"
    ) + $armArgs + @("--trace-dir", $trace, "--out", $out))
    Invoke-Python @commandArgs
    Invoke-Python tools/split_eval.py $out
}

switch ($Dataset) {
    "rootse" { Run-RootSE }
    "sp" { Run-SP }
    "all" { Run-RootSE; Run-SP }
}

Write-Host "`nResults: $RunDir" -ForegroundColor Green

