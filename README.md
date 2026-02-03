# MathSolver

**Offline equation solver with OCR and symbolic computation for Linux.**

Capture equations via screenshot or LaTeX input, get step-by-step symbolic solutions. Optimized for physics and computational math coursework. Zero network calls, zero AI querying - pure algorithmic routing.

## Features

- 📷 **Screenshot Capture**: Select any equation on screen, auto-detect via OCR
- ⌨️ **Direct Input**: LaTeX or plain text equation entry
- 🔬 **Smart Classification**: Routes equations to specialized solvers
- 📝 **Step-by-Step Solutions**: Pedagogical output showing derivation
- 🔢 **Numerical Evaluation**: Plug in values with physical constants library
- 💾 **History Database**: Track and search past solutions
- 🐧 **Linux Native**: Supports GNOME, KDE, i3/sway

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mathsolver.git
cd mathsolver

# Create virtual environment (Python 3.10-3.12 recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### OCR Support (Optional)

Screenshot-to-LaTeX OCR requires pix2tex, which has specific version requirements:

```bash
# Requires Python 3.10-3.12 (not 3.13+ due to NumPy 2.0 incompatibility)
pip install 'pix2tex[gui]'
```

**Note**: pix2tex is optional. The solver works without OCR - just enter equations directly via LaTeX or plain text.

### CLI Usage

```bash
# Launch GUI
python main.py

# Solve equation in terminal
python main.py "x^2 - 4 = 0"

# Solve with steps
python main.py -s "x^2 + 2x + 1 = 0"

# Solve for specific variable with numerical values  
python main.py -v E -n m=5 -n c=299792458 "E = mc^2"

# Output formats
python main.py -f latex "F = ma"    # LaTeX document
python main.py -f python "y = mx"   # Python code
python main.py -f json "x^2 = 4"    # JSON output

# List/search physics formulas
python main.py --list-formulas
python main.py --search "energy"
```

## Supported Equation Types

- ✅ General algebraic equations
- ✅ Polynomial equations  
- ✅ Physics formulas (62 templates)
- ✅ Ordinary differential equations (ODEs)
- ✅ Calculus (derivatives, integrals, limits, series)
- ✅ Trigonometric equations

## Dependencies

**Python:** 3.10+ (3.10-3.12 for OCR support)

**Core Libraries:**
- `sympy` - Symbolic mathematics
- `PyQt6` - GUI framework
- `Pillow` - Image handling
- `antlr4-python3-runtime` - LaTeX parsing
- `pint` - Physical units

**Optional:**
- `pix2tex` - LaTeX OCR (requires Python ≤3.12)

## Building Standalone Executable

```bash
pip install pyinstaller
pyinstaller mathsolver.spec
./dist/mathsolver/mathsolver --help
```

## Security

- Input validation prevents code injection
- Parameterized SQL queries
- No shell=True in subprocess calls

## License

MIT License
