<#
.SYNOPSIS
    One-click PowerShell launcher that (1) downloads the Roboflow public licence-plate dataset if missing and (2) kicks off Ultralytics YOLOv8 fine-tuning.

.DESCRIPTION
    • If the dataset folder already exists, skipping the download step saves time.
    • You can override key hyper-parameters with script arguments (see PARAMS).
    • Requires:   - Python 3.8+ with `ultralytics` package   - 7-Zip (or native `Expand-Archive`) for extraction.

.PARAMETER epochs
    Number of training epochs (default = 60)
.PARAMETER model
    Base model to fine-tune (default = "yolov8s.pt"). Accepts yolov8n.pt, yolov8m.pt, ...
.PARAMETER imgsz
    Training image size (default = 640)  

.EXAMPLE
    # Run with defaults
    ./train_plates.ps1

.EXAMPLE
    # Custom epochs and nano model to speed up on CPU
    ./train_plates.ps1 -epochs 30 -model "yolov8n.pt"
#>

param(
    [int]$epochs = 60,
    [string]$model = "yolov8s.pt",
    [int]$imgsz = 640
)

$ErrorActionPreference = 'Stop'

Write-Host "=== YOLOv8 Plate-training Launcher ===" -ForegroundColor Cyan

# ---------------------------
# 1. Dataset download / check
# ---------------------------
$zipPath = "plates.zip"
$datasetDir = "dataset"

if (-not (Test-Path $datasetDir)) {
    if (-not (Test-Path $zipPath)) {
        Write-Error "plates.zip not found. Please place your dataset zip in the project root and rerun the script."
        exit 1
    }
    Write-Host "Extracting dataset..." -ForegroundColor Yellow
    Write-Host "Extracting dataset..." -ForegroundColor Yellow
    Expand-Archive -Path $zipPath -DestinationPath $datasetDir
    Remove-Item $zipPath
} else {
    Write-Host "Dataset folder already exists - skipping download." -ForegroundColor Green
}

# ---------------------------
# 2. Launch training
# ---------------------------
Write-Host "Starting Ultralytics training..." -ForegroundColor Cyan
# Determine yolo command
$yoloCmd = "yolo"
try {
    Get-Command $yoloCmd -ErrorAction Stop | Out-Null
} catch {
    $alt = Join-Path (Join-Path $env:USERPROFILE "AppData\\Roaming\\Python\\Python312\\Scripts") "yolo.exe"
    if (Test-Path $alt) {
        $yoloCmd = $alt
    } else {
        $yoloCmd = "python -m ultralytics"
    }
}

# Build training argument array
$trainArgs = @(
    "task=detect",
    "mode=train",
    "model=$model",
    "data=$datasetDir/data.yaml",
    "epochs=$epochs",
    "imgsz=$imgsz",
    "hsv_h=0.015","hsv_s=0.7","hsv_v=0.4",
    "degrees=2","translate=0.1","scale=0.4","shear=2",
    "mosaic=1.0","perspective=0.0","mixup=0.1"
)

# Echo full command for visibility
Write-Host "$yoloCmd $($trainArgs -join ' ')" -ForegroundColor Gray

# Execute using call operator so quoting is handled properly
& $yoloCmd @trainArgs


Write-Host "`nTraining finished. Best weights: runs/detect/train/weights/best.pt" -ForegroundColor Green
