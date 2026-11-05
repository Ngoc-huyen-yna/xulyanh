$ErrorActionPreference = "Stop"

$variants = @("original", "contrast", "clahe", "denoise", "sharpen", "combo")

Write-Host "=== TV3: extracting features for all image variants ==="
foreach ($variant in $variants) {
    Write-Host "`n--- Extracting: $variant ---"
    & python (Join-Path $PSScriptRoot "tv3_extract_features.py") --variant $variant
    if ($LASTEXITCODE -ne 0) {
        throw "Feature extraction failed for '$variant' (exit code $LASTEXITCODE)."
    }
}

Write-Host "`n=== TV3: training all variants ==="
foreach ($variant in $variants) {
    Write-Host "`n--- Training: $variant ---"
    & python (Join-Path $PSScriptRoot "tv3_train_classic.py") --variant $variant
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed for '$variant' (exit code $LASTEXITCODE)."
    }
}

Write-Host "`nTV3 pipeline completed. Results: $(Join-Path $PSScriptRoot 'results_tv3.csv')"
