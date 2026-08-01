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
    @{ Source = 'operational_stage2_joint_separated_capability_marginal_v1\run_manifest.json'; Destination = 'operational_stage2\run_manifest.json' },
    @{ Source = 'joint_response_separated_capability_marginal_v1\run_manifest.json'; Destination = 'sensitivity\graded_response\run_manifest.json' },
    @{ Source = 'joint_response_separated_capability_marginal_v1\status.json'; Destination = 'sensitivity\graded_response\status.json' },
    @{ Source = 'joint_response_separated_capability_marginal_v1\runtime_summary.json'; Destination = 'sensitivity\graded_response\runtime_summary.json' }
) | ForEach-Object {
    Copy-ReleaseFile -RelativeSource $_.Source -RelativeDestination $_.Destination
}

Copy-ReleaseTree -RelativeSource 'tables' -RelativeDestination 'base_tables' -Extensions @('.csv', '.tex')
Copy-ReleaseTree -RelativeSource 'figures\maps' -RelativeDestination 'figures\maps' -Extensions @('.png', '.pdf', '.svg', '.md', '.json')
Copy-ReleaseTree -RelativeSource 'correlated_facility_separated_capability_marginal_v1\tables' -RelativeDestination 'correlated_facility\tables' -Extensions @('.csv', '.tex')
Copy-ReleaseTree -RelativeSource 'correlated_facility_separated_capability_marginal_v1\configs' -RelativeDestination 'correlated_facility\configs' -Extensions @('.json')
Copy-ReleaseTree -RelativeSource 'operational_stage2_joint_separated_capability_marginal_v1\tables' -RelativeDestination 'operational_stage2\tables' -Extensions @('.csv', '.tex')
Copy-ReleaseTree -RelativeSource 'joint_sensitivity_separated_capability_marginal_v1\tables' -RelativeDestination 'sensitivity\density_cap\tables' -Extensions @('.csv', '.tex')
Copy-ReleaseTree -RelativeSource 'joint_sensitivity_separated_capability_marginal_v1\selected_diagnostics' -RelativeDestination 'sensitivity\density_cap\selected_diagnostics' -Extensions @('.json', '.csv')
Copy-ReleaseTree -RelativeSource 'joint_response_separated_capability_marginal_v1\tables' -RelativeDestination 'sensitivity\graded_response\tables' -Extensions @('.csv', '.tex')
Copy-ReleaseTree -RelativeSource 'joint_response_separated_capability_marginal_v1\selected_diagnostics' -RelativeDestination 'sensitivity\graded_response\selected_diagnostics' -Extensions @('.json', '.csv')

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

