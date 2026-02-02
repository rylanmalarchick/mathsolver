# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for MathSolver.

Build with:
    pyinstaller mathsolver.spec

This creates a single-folder distribution in dist/mathsolver/
"""

import os
import sys
from pathlib import Path

# Get the project root
PROJECT_ROOT = Path(SPECPATH).resolve()

# Analysis configuration
a = Analysis(
    ['main.py'],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=[
        # Include config files
        ('config', 'config'),
        # Include data directory structure (empty, but needed)
        ('data', 'data'),
    ],
    hiddenimports=[
        # SymPy modules
        'sympy',
        'sympy.parsing.latex',
        'sympy.parsing.sympy_parser',
        'sympy.physics.units',
        'sympy.polys',
        'sympy.solvers',
        'sympy.integrals',
        'sympy.series',
        # NumPy/SciPy
        'numpy',
        'scipy',
        'scipy.special',
        # PyQt6
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtWidgets',
        'PyQt6.QtGui',
        'PyQt6.sip',
        # Optional: WebEngine (comment out if not using MathJax rendering)
        # 'PyQt6.QtWebEngineWidgets',
        # 'PyQt6.QtWebChannel',
        # PIL
        'PIL',
        'PIL.Image',
        # Pint units
        'pint',
        # ANTLR for LaTeX parsing
        'antlr4',
        'antlr4.error',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude pix2tex/torch by default (too large, ~2GB)
        # Users can install OCR separately
        'pix2tex',
        'torch',
        'torchvision',
        'tensorflow',
        'keras',
        # Exclude unused Qt modules
        'PyQt6.QtBluetooth',
        'PyQt6.QtDBus',
        'PyQt6.QtDesigner',
        'PyQt6.QtMultimedia',
        'PyQt6.QtNetwork',
        'PyQt6.QtNfc',
        'PyQt6.QtQml',
        'PyQt6.QtQuick',
        'PyQt6.QtSensors',
        'PyQt6.QtSerialPort',
        'PyQt6.QtSql',
        'PyQt6.QtTest',
        'PyQt6.QtXml',
    ],
    noarchive=False,
    optimize=0,
)

# Create PYZ archive
pyz = PYZ(a.pure)

# Create executable
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='mathsolver',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI application, no console
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # TODO: Add icon path if available
)

# Collect all files into distribution folder
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='mathsolver',
)
