# ============================================================
#  TripMate — One-shot Setup Script
#
#  Usage (run once from the project root):
#    .\setup.ps1
#
#  What it does:
#    1. Creates a Python virtual environment (venv/)
#    2. Installs all dependencies from requirements.txt
#    3. Installs msgpack (pure-Python msgpack backend)
#    4. Patches ormsgpack with a pure-Python shim so the app
#       works even when Windows Application Control blocks .pyd DLLs
#    5. Verifies everything is working
# ============================================================

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$VenvPath    = Join-Path $ProjectRoot "venv"
$PipExe      = Join-Path $VenvPath "Scripts\pip.exe"
$PythonExe   = Join-Path $VenvPath "Scripts\python.exe"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  TripMate - Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Create virtual environment ─────────────────────
Write-Host "[1/4] Creating virtual environment..." -ForegroundColor Yellow
python -m venv $VenvPath
Write-Host "      OK" -ForegroundColor Green

# ── Step 2: Install project dependencies ───────────────────
Write-Host "[2/4] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
& $PipExe install -r (Join-Path $ProjectRoot "requirements.txt") --quiet
Write-Host "      OK" -ForegroundColor Green

# ── Step 3: Install pure-Python msgpack (shim backend) ─────
Write-Host "[3/4] Installing msgpack (shim backend)..." -ForegroundColor Yellow
& $PipExe install msgpack --quiet
Write-Host "      OK" -ForegroundColor Green

# ── Step 4: Patch ormsgpack with pure-Python shim ──────────
Write-Host "[4/4] Patching ormsgpack (Application Control bypass)..." -ForegroundColor Yellow

$ShimSource = Join-Path $ProjectRoot "_ormsgpack_shim.py"
$ShimDest   = Join-Path $VenvPath "Lib\site-packages\ormsgpack\__init__.py"

if (-not (Test-Path $ShimSource)) {
    Write-Host "      ERROR: _ormsgpack_shim.py not found in project root!" -ForegroundColor Red
    exit 1
}

Copy-Item -Path $ShimSource -Destination $ShimDest -Force
Write-Host "      Shim copied to $ShimDest" -ForegroundColor Green

# ── Verify ─────────────────────────────────────────────────
$TestOutput = & $PythonExe -c "import ormsgpack; d=ormsgpack.packb({'ok':True}); print(ormsgpack.unpackb(d))" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "      Verification: $TestOutput" -ForegroundColor Green
} else {
    Write-Host "      Verification FAILED: $TestOutput" -ForegroundColor Red
    exit 1
}

# ── Done ───────────────────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Setup complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor White
Write-Host ""
Write-Host "    1. Fill in your API keys in .env" -ForegroundColor Gray
Write-Host ""
Write-Host "    2. Activate the venv:" -ForegroundColor Gray
Write-Host "       venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "    3. Run the app:" -ForegroundColor Gray
Write-Host "       python app.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "    4. Open in browser:" -ForegroundColor Gray
Write-Host "       http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host ""
