#!/bin/bash
# MathSolver Installation Script
# Tested on: Ubuntu 22.04, Arch Linux, Fedora 38

set -e

echo "╔════════════════════════════════════════════════╗"
echo "║          MathSolver Installation               ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# Detect package manager
if command -v apt &> /dev/null; then
    PKG_MANAGER="apt"
    echo "[*] Detected: Debian/Ubuntu"
elif command -v pacman &> /dev/null; then
    PKG_MANAGER="pacman"
    echo "[*] Detected: Arch Linux"
elif command -v dnf &> /dev/null; then
    PKG_MANAGER="dnf"
    echo "[*] Detected: Fedora"
else
    echo "[!] Unknown package manager. Please install dependencies manually."
    PKG_MANAGER="unknown"
fi

# Install system dependencies
echo ""
echo "[1/5] Installing system dependencies..."

if [ "$PKG_MANAGER" = "apt" ]; then
    echo "      Running: sudo apt install python3-venv python3-pip ..."
    sudo apt update
    sudo apt install -y python3-venv python3-pip python3-dev \
                        libxcb-cursor0 libxkbcommon0 \
                        gnome-screenshot || true  # screenshot tool optional
elif [ "$PKG_MANAGER" = "pacman" ]; then
    echo "      Running: sudo pacman -S python python-pip ..."
    sudo pacman -S --noconfirm python python-pip \
                               gnome-screenshot || true
elif [ "$PKG_MANAGER" = "dnf" ]; then
    echo "      Running: sudo dnf install python3-devel ..."
    sudo dnf install -y python3-devel python3-pip \
                        gnome-screenshot || true
fi

# Check for screenshot tool
echo ""
echo "[2/5] Checking screenshot tools..."
SCREENSHOT_TOOL=""
for tool in gnome-screenshot spectacle maim scrot; do
    if command -v $tool &> /dev/null; then
        SCREENSHOT_TOOL=$tool
        echo "      Found: $tool ✓"
        break
    fi
done

if [ -z "$SCREENSHOT_TOOL" ]; then
    echo "      [!] No screenshot tool found. Install one of:"
    echo "          gnome-screenshot, spectacle, maim, or scrot"
    echo "          Screenshot feature will not work without one."
fi

# Create virtual environment
echo ""
echo "[3/5] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU-only for smaller install)
echo ""
echo "[4/5] Installing PyTorch (CPU-only, this may take a while)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo ""
echo "[5/5] Installing Python dependencies..."
pip install -r requirements.txt

# Download pix2tex model (first run)
echo ""
echo "[*] Downloading pix2tex OCR model (~500MB)..."
echo "    This happens on first import..."
python -c "
try:
    from pix2tex.cli import LatexOCR
    print('    Model loaded successfully!')
except Exception as e:
    print(f'    Note: Model will download on first use. ({e})')
"

# Create data directory
mkdir -p data

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║          Installation Complete!                ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "To run MathSolver:"
echo ""
echo "    source venv/bin/activate"
echo "    python main.py"
echo ""
echo "Or create a desktop shortcut:"
echo ""
echo "    cp mathsolver.desktop ~/.local/share/applications/"
echo ""
