[CmdletBinding()]
param(
    [string]$SourceRoot,
    [string]$OutputRoot
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot

if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
    $SourceRoot = Join-Path $repositoryRoot 'data_work\noto\acute_access_graded_v4'
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $repositoryRoot 'results\noto\acute_access_graded_v4'
}

$sourcePath = (Resolve-Path -LiteralPath $SourceRoot).Path
$outputPath = [System.IO.Path]::GetFullPath($OutputRoot)
if (Test-Path -LiteralPath $outputPath) {
    Remove-Item -LiteralPath $outputPath -Recurse -Force
}


if (-not (Test-Path -LiteralPath (Join-Path $sourcePath 'run_design.json'))) {
    throw "Expected frozen Noto run outputs under $sourcePath."
}

New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

function Copy-ReleaseFile {
    param(
        [Parameter(Mandatory = $true)][string]$RelativeSource,
        [Parameter(Mandatory = $true)][string]$RelativeDestination
    )

    $sourceFile = Join-Path $sourcePath $RelativeSource
    if (-not (Test-Path -LiteralPath $sourceFile -PathType Leaf)) {
        throw "Missing release artifact: $sourceFile"
    }

    $destinationFile = Join-Path $outputPath $RelativeDestination
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destinationFile) | Out-Null
    Copy-Item -LiteralPath $sourceFile -Destination $destinationFile -Force
}

function Copy-ReleaseTree {
    param(
        [Parameter(Mandatory = $true)][string]$RelativeSource,
        [Parameter(Mandatory = $true)][string]$RelativeDestination,
        [Parameter(Mandatory = $true)][string[]]$Extensions
    )

    $sourceDirectory = Join-Path $sourcePath $RelativeSource
    if (-not (Test-Path -LiteralPath $sourceDirectory -PathType Container)) {
        throw "Missing release directory: $sourceDirectory"
    }

    $destinationDirectory = Join-Path $outputPath $RelativeDestination
    Get-ChildItem -LiteralPath $sourceDirectory -File -Recurse |
        Where-Object { $Extensions -contains $_.Extension.ToLowerInvariant() } |
        ForEach-Object {
            $relativePath = $_.FullName.Substring($sourceDirectory.Length).TrimStart([char[]]@('\', '/'))
            $destinationFile = Join-Path $destinationDirectory $relativePath
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destinationFile) | Out-Null
            Copy-Item -LiteralPath $_.FullName -Destination $destinationFile -Force
        }
}

@(
    @{ Source = 'run_design.json'; Destination = 'frozen_inputs\run_design.json' },
    @{ Source = 'reproducibility.json'; Destination = 'frozen_inputs\reproducibility.json' },
    @{ Source = 'practitioner_layer_design.json'; Destination = 'frozen_inputs\practitioner_layer_design.json' },
    @{ Source = 'runtime_summary.json'; Destination = 'frozen_inputs\runtime_summary.json' },
    @{ Source = 'certificate_runtime_summary.json'; Destination = 'frozen_inputs\certificate_runtime_summary.json' },
    @{ Source = 'operational_stage2_joint_separated_capability_marginal_v2\run_manifest.json'; Destination = 'operational_stage2\run_manifest.json' }
) | ForEach-Object {
    Copy-ReleaseFile -RelativeSource $_.Source -RelativeDestination $_.Destination
}

Copy-ReleaseTree -RelativeSource 'tables' -RelativeDestination 'base_tables' -Extensions @('.csv', '.tex')
Copy-ReleaseTree -RelativeSource 'figures\maps' -RelativeDestination 'figures\maps' -Extensions @('.png', '.pdf', '.svg', '.md', '.json')
Copy-ReleaseTree -RelativeSource 'correlated_facility_separated_capability_marginal_v2\tables' -RelativeDestination 'correlated_facility\tables' -Extensions @('.csv', '.tex')
Copy-ReleaseTree -RelativeSource 'operational_stage2_joint_separated_capability_marginal_v2\tables' -RelativeDestination 'operational_stage2\tables' -Extensions @('.csv', '.tex')
Copy-ReleaseTree -RelativeSource 'joint_sensitivity_separated_capability_marginal_v1\tables' -RelativeDestination 'sensitivity\density_cap\tables' -Extensions @('.csv', '.tex')

$checksumFile = Join-Path $outputPath 'SHA256SUMS.txt'
$hashLines = Get-ChildItem -LiteralPath $outputPath -File -Recurse |
    Where-Object { $_.FullName -ne $checksumFile } |
    Sort-Object FullName |
    ForEach-Object {
        $relativePath = $_.FullName.Substring($outputPath.Length).TrimStart([char[]]@('\', '/')).Replace('\', '/')
        "$(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256 | Select-Object -ExpandProperty Hash) *$relativePath"
    }

Set-Content -LiteralPath $checksumFile -Value $hashLines -Encoding utf8
Write-Output "Public release artifacts written to $outputPath"

$legacyDirectory = Join-Path $outputPath "base_tables"
if (Test-Path -LiteralPath $legacyDirectory) {
    Remove-Item -LiteralPath $legacyDirectory -Recurse -Force
}

$auditDirectory = Join-Path $outputPath "audits"
New-Item -ItemType Directory -Force -Path $auditDirectory | Out-Null
foreach ($auditFile in @(
    "master_manifest.json",
    "final_test_output.txt",
    "table_monotonicity_audit.csv",
    "table_nominal_marginal_audit.csv",
    "table_worst_case_marginal_audit.csv",
    "table_numerical_replay_audit.csv",
    "table_probability_audit.csv",
    "table_runtime_checkpoint_audit.csv"
)) {
    Copy-Item -LiteralPath (Join-Path $sourcePath "final_computational_audit_v1/$auditFile") -Destination (Join-Path $auditDirectory $auditFile) -Force
}

$certificateTable = Join-Path $sourcePath "correlated_facility_separated_capability_marginal_v2/tables/table_noto_correlated_continuous_certificate.csv"
$certificateRows = Import-Csv -LiteralPath $certificateTable
$completion = [ordered]@{
    status = "completed"
    certificate_table = "correlated_facility/tables/table_noto_correlated_continuous_certificate.csv"
    rho_values = @($certificateRows | ForEach-Object { [double]$_.rho })
    cells_per_radius = [int]$certificateRows[0].cell_count
    method = "budget-intersecting grid cells evaluated at monotonic upper corners"
    checkpoints = "excluded from public release; resumable locally"
}
$completion | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $outputPath "frozen_inputs/continuous_certificate_completion.json") -Encoding utf8

$revision = (& git -C $repositoryRoot rev-parse HEAD).Trim()
$releaseManifest = [ordered]@{
    release_version = "noto-corrected-v2"
    generated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    code_revision = $revision
    stage1_source = "correlated_facility_separated_capability_marginal_v2"
    state_count = 128
    candidate_policy_count = 996
    legacy_independent_state_tables = "excluded from this paper-ready release"
}
$releaseManifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $outputPath "frozen_inputs/public_release_manifest.json") -Encoding utf8

$readme = @(
    "# Corrected Noto Paper Results",
    "",
    "This is the paper-ready release for the corrected 128-state joint road-facility model.",
    "",
    "The package contains corrected formulation-specific Stage-1, continuous-certificate, mechanism, Stage-2, sensitivity, audit, and figure outputs.",
    "The release excludes the older independent 32-state tables.",
    "Raw source layers, solver checkpoints, and execution logs remain excluded.",
    "Frozen inputs, audit manifests, source hashes, and final-test output are included for reproducibility."
)
Set-Content -LiteralPath (Join-Path $outputPath "README.md") -Value $readme -Encoding utf8

$checksumFile = Join-Path $outputPath "SHA256SUMS.txt"
$hashLines = Get-ChildItem -LiteralPath $outputPath -File -Recurse |
    Where-Object { $_.FullName -ne $checksumFile } |
    Sort-Object FullName |
    ForEach-Object {
        $relativePath = $_.FullName.Substring($outputPath.Length).TrimStart([char[]]@([char](92), "/")).Replace([char]92, "/")
        "$(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256 | Select-Object -ExpandProperty Hash) *$relativePath"
    }
Set-Content -LiteralPath $checksumFile -Value $hashLines -Encoding utf8

$gradedDirectory = Join-Path $outputPath "sensitivity/graded_response"
New-Item -ItemType Directory -Force -Path (Join-Path $gradedDirectory "tables") | Out-Null
Copy-Item -LiteralPath (Join-Path $sourcePath "joint_sensitivity_separated_capability_marginal_v1/run_manifest.json") -Destination (Join-Path $gradedDirectory "run_manifest.json") -Force
Copy-Item -LiteralPath (Join-Path $sourcePath "joint_sensitivity_separated_capability_marginal_v1/status.json") -Destination (Join-Path $gradedDirectory "status.json") -Force
Copy-Item -LiteralPath (Join-Path $sourcePath "joint_sensitivity_separated_capability_marginal_v1/runtime_summary.json") -Destination (Join-Path $gradedDirectory "runtime_summary.json") -Force
Copy-Item -LiteralPath (Join-Path $sourcePath "joint_sensitivity_separated_capability_marginal_v1/tables/table_noto_graded_response_sensitivity.csv") -Destination (Join-Path $gradedDirectory "tables/table_noto_graded_response_sensitivity.csv") -Force
Copy-Item -LiteralPath (Join-Path $sourcePath "joint_sensitivity_separated_capability_marginal_v1/tables/table_noto_joint_sensitivity_full_grid.csv") -Destination (Join-Path $gradedDirectory "tables/table_noto_joint_sensitivity_full_grid.csv") -Force

$checksumFile = Join-Path $outputPath "SHA256SUMS.txt"
$hashLines = Get-ChildItem -LiteralPath $outputPath -File -Recurse |
    Where-Object { $_.FullName -ne $checksumFile } |
    Sort-Object FullName |
    ForEach-Object {
        $relativePath = $_.FullName.Substring($outputPath.Length).TrimStart([char[]]@([char](92), "/")).Replace([char]92, "/")
        "$(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256 | Select-Object -ExpandProperty Hash) *$relativePath"
    }
Set-Content -LiteralPath $checksumFile -Value $hashLines -Encoding utf8



Copy-ReleaseTree -RelativeSource "selected_sensitivity_separated_capability_marginal_v1/tables" -RelativeDestination "sensitivity/selected/tables" -Extensions @(".csv", ".tex")
Copy-ReleaseFile -RelativeSource "selected_sensitivity_separated_capability_marginal_v1/run_manifest.json" -RelativeDestination "sensitivity/selected/run_manifest.json"
Copy-ReleaseFile -RelativeSource "selected_sensitivity_separated_capability_marginal_v1/status.json" -RelativeDestination "sensitivity/selected/status.json"

$checksumFile = Join-Path $outputPath "SHA256SUMS.txt"
$hashLines = Get-ChildItem -LiteralPath $outputPath -File -Recurse |
    Where-Object { $_.FullName -ne $checksumFile } |
    Sort-Object FullName |
    ForEach-Object {
        $relativePath = $_.FullName.Substring($outputPath.Length).TrimStart([char[]]@([char](92), "/")).Replace([char](92), "/")
        "$(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256 | Select-Object -ExpandProperty Hash) *$relativePath"
    }
Set-Content -LiteralPath $checksumFile -Value $hashLines -Encoding utf8
