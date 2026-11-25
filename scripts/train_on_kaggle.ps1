# Pushes the training kernel to Kaggle, waits for the T4 GPU run to finish,
# and downloads the trained model into models/.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$kernelDir = Join-Path $root 'kaggle_kernel'
$kernel = 'abdullahasghar12/digit-gpu-trainer'
$modelsDir = Join-Path $root 'models'

kaggle kernels push -p $kernelDir
if ($LASTEXITCODE -ne 0) { throw 'kaggle kernels push failed' }

$status = ''
for ($i = 0; $i -lt 160; $i++) {
    Start-Sleep -Seconds 15
    $status = (kaggle kernels status $kernel 2>&1 | Out-String)
    Write-Output "Kaggle status: $($status.Trim())"
    if ($status -match 'COMPLETE') { break }
    if ($status -match 'ERROR|FAILED|CANCEL') { throw "Kaggle run failed: $status" }
}
if ($status -notmatch 'COMPLETE') { throw 'Timed out waiting for the Kaggle GPU run' }

New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null
kaggle kernels output $kernel -p $modelsDir
if ($LASTEXITCODE -ne 0) { throw 'kaggle kernels output failed' }
if (-not (Test-Path (Join-Path $modelsDir 'model.pt'))) { throw 'model.pt was not downloaded' }
Write-Output 'Trained model downloaded from Kaggle GPU run.'
