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

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Screenshot  │  │    LaTeX     │  │  Plain Text  │           │
│  │   Capture    │  │    Input     │  │    Input     │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                 │                 │                    │
│         ▼                 │                 │                    │
│  ┌──────────────┐         │                 │                    │
│  │  pix2tex     │         │                 │                    │
│  │  LaTeX-OCR   │─────────┴─────────────────┘                    │
│  └──────┬───────┘                                                │
└─────────┼────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        PARSING LAYER                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              SymPy LaTeX Parser (parse_latex)             │   │
│  │                                                           │   │
│  │   LaTeX String  ──────►  SymPy Expression Tree           │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────┬────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CLASSIFICATION LAYER                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  EquationClassifier                       │   │
│  │                                                           │   │
│  │   Priority: Physics ► ODE ► Calculus ► LinAlg ► General  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────┬────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        SOLVER LAYER                              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐   │
│  │  Physics   │ │    ODE     │ │  Calculus  │ │  General   │   │
│  │  Solver    │ │   Solver   │ │   Solver   │ │  Solver    │   │
│  │ (patterns) │ │  (dsolve)  │ │(diff/integ)│ │  (solve)   │   │
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘   │
│        └──────────────┴──────────────┴──────────────┘           │
└─────────┬────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   MathJax/   │  │    Step      │  │   Export     │           │
│  │  matplotlib  │  │  Generator   │  │   (LaTeX/    │           │
│  │  Rendering   │  │              │  │   Python)    │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└──────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mathsolver.git
cd mathsolver

# Run installation script
chmod +x install.sh
./install.sh

# Activate virtual environment
source venv/bin/activate

# Run the application
python main.py
```

### Usage

1. **Screenshot Mode**: Click "📷 Screenshot" → Select equation area → Solution appears
2. **LaTeX Mode**: Type LaTeX directly (e.g., `\frac{d}{dx} x^2 = 2x`)
3. **Plain Text Mode**: Type natural notation (e.g., `E = mc^2`)

## Project Structure

```
mathsolver/
├── main.py                      # Entry point
├── requirements.txt             # Python dependencies
├── install.sh                   # Installation script
├── README.md                    # This file
├── WORKPLAN.md                  # Development roadmap
│
├── config/
│   ├── settings.yaml            # User preferences
│   └── physics_formulas.json    # Physics formula database
│
├── src/
│   ├── __init__.py
│   ├── models.py                # Core data structures
│   │
│   ├── input/                   # Input layer
│   │   ├── screenshot.py        # Screenshot capture
│   │   ├── ocr.py               # pix2tex wrapper
│   │   └── parser.py            # LaTeX → SymPy
│   │
│   ├── classification/          # Classification layer
│   │   └── classifier.py        # Equation type detection
│   │
│   ├── solvers/                 # Solver layer
│   │   ├── base.py              # Abstract solver interface
│   │   ├── general.py           # General SymPy solver
│   │   ├── physics_solver.py    # Physics formulas (Week 2)
│   │   ├── ode_solver.py        # Differential equations
│   │   └── calculus_solver.py   # Derivatives/integrals
│   │
│   ├── output/                  # Output layer
│   │   ├── renderer.py          # Math rendering
│   │   └── step_generator.py    # Step-by-step text
│   │
│   ├── gui/                     # GUI layer
│   │   └── main_window.py       # PyQt6 main window
│   │
│   └── utils/                   # Utilities
│       ├── constants.py         # Physical constants
│       └── database.py          # SQLite history
│
├── tests/                       # Test suite
│   └── __init__.py
│
└── data/
    └── history.db               # Solution history
```

## Dependencies

**Python:** 3.10+

**Core Libraries:**
- `pix2tex[gui]` - LaTeX OCR (ViT + Transformer model)
- `sympy` - Symbolic mathematics
- `PyQt6` - GUI framework
- `Pillow` - Image handling

**Optional:**
- `PyQt6-WebEngine` - MathJax rendering
- `pint` - Unit conversion
- `scipy` - Numerical fallback

## Development Roadmap

See [WORKPLAN.md](WORKPLAN.md) for detailed weekly milestones.

| Week | Focus | Status |
|------|-------|--------|
| 1 | Core pipeline + basic GUI | ✅ Done |
| 2 | Physics pattern library + PhysicsSolver | ✅ Done |
| 3 | Step-by-step generation + MathJax | ✅ Done |
| 4 | Polish, testing, packaging | ✅ Done |

## Supported Equation Types

### Implemented
- ✅ General algebraic equations
- ✅ Polynomial equations  
- ✅ Physics formulas (62 templates)
- ✅ Ordinary differential equations (ODEs)
- ✅ Calculus (derivatives, integrals, limits, series)
- ✅ Trigonometric equations

### CLI Usage

```bash
# Launch GUI
mathsolver

# Solve equation in terminal
mathsolver "x^2 - 4 = 0"

# Solve with steps
mathsolver -s "x^2 + 2x + 1 = 0"

# Solve for specific variable with numerical values
mathsolver -v E -n m=5 -n c=299792458 "E = mc^2"

# Output formats
mathsolver -f latex "F = ma"    # LaTeX document
mathsolver -f python "y = mx"   # Python code
mathsolver -f json "x^2 = 4"    # JSON output

# List/search physics formulas
mathsolver --list-formulas
mathsolver --search "energy"
```

## Performance Targets

| Component | Target |
|-----------|--------|
| OCR Processing | 200-500ms |
| LaTeX Parsing | <50ms |
| Classification | <10ms |
| Symbolic Solve | 10ms-1s |
| Rendering | <100ms |
| **Total E2E** | **<2 seconds** |

## Contributing

This is a personal project for coursework, but feel free to fork and adapt for your own use.

## License

MIT License
