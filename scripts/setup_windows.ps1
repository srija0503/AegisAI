# setup_windows.ps1
# ══════════════════════════════════════════════════════
#  First-time setup script for Windows (PowerShell)
# ══════════════════════════════════════════════════════

Write-Host ""
Write-Host "╔══════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Quantum-Secure Federated Firewall Setup     ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── Check Python version ──────────────────────────────
$py = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Python not found. Install Python 3.10+ from https://python.org" -ForegroundColor Red
    exit 1
}
Write-Host "✓ $py detected" -ForegroundColor Green

# ── Create virtual environment ────────────────────────
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
}

# ── Activate venv ─────────────────────────────────────
. ".venv\Scripts\Activate.ps1"
Write-Host "✓ Virtual environment activated" -ForegroundColor Green

# ── Upgrade pip ───────────────────────────────────────
python -m pip install --upgrade pip -q
Write-Host "✓ pip upgraded" -ForegroundColor Green

# ── Install dependencies ──────────────────────────────
Write-Host "Installing dependencies (this may take several minutes)..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Dependency installation failed." -ForegroundColor Red
    exit 1
}
Write-Host "✓ Dependencies installed" -ForegroundColor Green

# ── Copy .env ─────────────────────────────────────────
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "✓ .env created from .env.example — EDIT IT with your settings!" -ForegroundColor Yellow
} else {
    Write-Host "✓ .env already exists" -ForegroundColor Green
}

# ── Create required data directories ─────────────────
$dataDirs = @(
    "data\raw\network_traffic",
    "data\raw\intrusion",
    "data\raw\cve",
    "data\processed\traffic_features",
    "data\processed\rl_training",
    "data\processed\threat_intelligence",
    "data\vector_db",
    "data\sample",
    "rl_agent\models\checkpoints",
    "logs"
)
foreach ($d in $dataDirs) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}
Write-Host "✓ Data directories created" -ForegroundColor Green

Write-Host ""
Write-Host "══════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Setup complete! Next steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  1. Edit .env with your config" -ForegroundColor White
Write-Host "  2. make data        # download datasets" -ForegroundColor White
Write-Host "  3. make preprocess  # preprocess data" -ForegroundColor White
Write-Host "  4. make train       # train RL agent" -ForegroundColor White
Write-Host "  5. make simulate    # run simulation" -ForegroundColor White
Write-Host "══════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
