# MathSolver Development Workplan

## Overview

4-week development plan (10-15 hours/week) to build a fully functional offline equation solver.

---

## Week 1: Core Pipeline + Basic GUI ✅

**Goal:** End-to-end working prototype with manual LaTeX input.

### Completed Tasks

- [x] Project structure and file organization
- [x] Core data structures (`models.py`)
  - `Equation`, `Solution`, `SolutionStep`, `SolveRequest` dataclasses
  - `EquationType` enum for classification
- [x] Screenshot capture module (`input/screenshot.py`)
  - Auto-detection of gnome-screenshot, spectacle, maim, scrot
  - Area selection and image capture
- [x] OCR wrapper (`input/ocr.py`)
  - pix2tex integration with lazy loading
  - Confidence estimation heuristics
- [x] LaTeX parser (`input/parser.py`)
  - SymPy `parse_latex` integration
  - OCR error correction patterns
  - Plain text parsing fallback
- [x] Equation classifier (`classification/classifier.py`)
  - Priority-based type detection
  - ODE, calculus, polynomial, general categories
- [x] Base solver interface (`solvers/base.py`)
  - `BaseSolver` abstract class
  - `SolverResult` wrapper
  - `SolverRegistry` for solver selection
- [x] General solver (`solvers/general.py`)
  - SymPy `solve()` wrapper
  - Basic step generation
  - Variable isolation
- [x] Output rendering (`output/renderer.py`)
  - Matplotlib mathtext backend
  - MathJax HTML generation
- [x] Step generator (`output/step_generator.py`)
  - Linear and quadratic equation steps
- [x] Physical constants library (`utils/constants.py`)
  - 20+ common constants with SI units
- [x] History database (`utils/database.py`)
  - SQLite storage for solved equations
- [x] Main GUI window (`gui/main_window.py`)
  - Input panel with LaTeX/plain text entry
  - Classification display
  - Variable selector
  - Solution display
  - Export buttons
- [x] Entry point (`main.py`)
- [x] Documentation (README.md, WORKPLAN.md)
- [x] Installation script and requirements

### Week 1 Deliverable

A working application that:
1. Accepts LaTeX or plain text input
2. Parses to SymPy expression
3. Classifies equation type
4. Solves symbolically with basic steps
5. Displays solution in GUI

---

## Week 2: Physics Pattern Library + PhysicsSolver

**Goal:** Custom physics formula recognition and solving with domain knowledge.

### Tasks

- [ ] **Physics Formula Database** (`config/physics_formulas.json`)
  - Wien's Displacement Law: λ_max * T = 2.898×10⁻³
  - Planck energy: E = hν
  - Energy-momentum: E² = (pc)² + (mc²)²
  - Kinematic equations (5 standard forms)
  - Optics: n₁sinθ₁ = n₂sinθ₂, d sinθ = mλ
  - Michelson interferometer: Δ = 2d(n-1)
  - Thermodynamics: PV = nRT, ΔS = Q/T
  - Electromagnetism: F = qE, F = qv×B
  - **Target: 50+ formulas**

- [ ] **Pattern Matching System** (`classification/physics_patterns.py`)
  - SymPy `Wild` symbol matching
  - Structure-based hash lookup
  - Fuzzy matching for near-matches
  - Confidence scoring

- [ ] **Physics Solver** (`solvers/physics_solver.py`)
  - Pre-computed solution templates
  - Variable isolation logic
  - Unit-aware solving (Pint integration)
  - Step generation with physical explanations

- [ ] **Unit Conversion** (`utils/units.py`)
  - Pint library integration
  - Auto-conversion between compatible units
  - Dimensionality checking

- [ ] **Numerical Evaluation Panel**
  - Input fields for known values
  - Auto-populate with constants
  - Unit conversion in evaluation

### Week 2 Deliverable

Solving physics problems like:
- "Given T = 5800K, find λ_max using Wien's Law"
- "E = mc², find E when m = 1kg"
- Step-by-step with physical explanations

---

## Week 3: Step-by-Step Generation + MathJax Rendering

**Goal:** Beautiful, pedagogical output with proper math rendering.

### Tasks

- [ ] **MathJax Integration**
  - QtWebEngine view for MathJax
  - Local MathJax bundle (offline)
  - LaTeX → rendered equation
  - Copy-paste preserves LaTeX

- [ ] **Advanced Step Generator**
  - Chain rule steps for derivatives
  - Integration by substitution
  - Polynomial factorization steps
  - Quadratic formula derivation
  - ODE method explanations

- [ ] **Collapsible Steps UI**
  - TreeWidget or custom collapsible panels
  - "Show all" vs "Final answer only" toggle
  - Syntax highlighting for operations

- [ ] **ODE Solver** (`solvers/ode_solver.py`)
  - SymPy `dsolve` wrapper
  - Classification display (separable, linear, exact, etc.)
  - Method hint selection
  - Boundary condition handling

- [ ] **Calculus Solver** (`solvers/calculus_solver.py`)
  - Derivative with chain rule
  - Integral with substitution hints
  - Limit evaluation
  - Series expansion

### Week 3 Deliverable

- Properly rendered equations in GUI
- Step-by-step for derivatives showing chain rule
- ODE solving with method explanation

---

## Week 4: Polish, Testing, Packaging

**Goal:** Production-ready application with full test coverage.

### Tasks

- [ ] **Testing Suite**
  - Unit tests for all solvers
  - Parser edge case tests
  - OCR accuracy tests with sample images
  - Integration tests for full pipeline
  - GUI tests with pytest-qt

- [ ] **History Feature**
  - History dialog with search
  - Re-solve from history
  - Export history to file

- [ ] **Error Handling**
  - Graceful OCR failures
  - Parser error suggestions
  - Timeout for long solves
  - User-friendly error messages

- [ ] **Screenshot Integration**
  - Global keybind setup (i3/sway/KDE/GNOME)
  - Tray icon option
  - Quick-solve mode

- [ ] **Export Features**
  - PDF export with rendered steps
  - Python/SymPy code export
  - LaTeX document export

- [ ] **Packaging**
  - PyInstaller single executable
  - Desktop entry file
  - Man page
  - Install script for /usr/local

- [ ] **Documentation**
  - User manual
  - Formula database format docs
  - Adding custom formulas guide

### Week 4 Deliverable

- 90%+ test coverage
- Packaged executable
- Desktop integration
- Complete documentation

---

## Future Enhancements (Post-v1.0)

### High Priority
- [ ] Linear algebra solver (matrices, systems, eigenvalues)
- [ ] Numerical fallback (scipy.optimize)
- [ ] Handwriting OCR fine-tuning

### Medium Priority
- [ ] Graphing capabilities (matplotlib integration)
- [ ] Multiple equation solving
- [ ] LaTeX document mode

### Low Priority
- [ ] Plugin system for custom solvers
- [ ] Mobile companion app
- [ ] Optional Wolfram Alpha fallback

---

## Technical Debt Tracker

| Item | Priority | Week |
|------|----------|------|
| Replace text solution display with MathJax | High | 3 |
| Add proper logging | Medium | 4 |
| Type hints for all functions | Low | 4 |
| Docstrings for public API | Medium | 4 |
| Configuration file support | Medium | 3 |

---

## Testing Checklist

### Equations to Test (Add from coursework)

**Physics:**
- [ ] Wien's Law: λT = 2.898×10⁻³
- [ ] Planck: E = hν
- [ ] Einstein: E = mc²
- [ ] de Broglie: λ = h/p
- [ ] Kinematic: v² = v₀² + 2ax

**Calculus:**
- [ ] d/dx(x²) = 2x
- [ ] ∫x²dx = x³/3
- [ ] d/dx(sin(x)) = cos(x)
- [ ] ∫₀^π sin(x)dx = 2

**ODEs:**
- [ ] dy/dx = y → y = Ce^x
- [ ] y'' + y = 0 → y = A cos(x) + B sin(x)

**Polynomials:**
- [ ] x² - 1 = 0 → x = ±1
- [ ] x³ - 6x² + 11x - 6 = 0 → x = 1, 2, 3

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| pix2tex accuracy issues | High | Manual correction field, confidence threshold |
| SymPy parsing failures | High | Fallback parsers, error suggestions |
| PyTorch model size (~500MB) | Medium | Lazy loading, consider smaller model |
| Qt WebEngine complexity | Medium | Matplotlib fallback renderer |
| Physics pattern scaling | Low | Hash-based lookup, lazy loading |
