"""
Physical constants library.

Standard SI values for physics calculations.
"""

import sympy as sp


# Physical constants as SymPy symbols with numerical values
# These can be substituted into expressions for numerical evaluation

PHYSICAL_CONSTANTS = {
    # Fundamental constants
    "c": {
        "value": 299792458,
        "unit": "m/s",
        "name": "Speed of light in vacuum",
        "symbol": sp.Symbol("c"),
    },
    "h": {
        "value": 6.62607015e-34,
        "unit": "J·s",
        "name": "Planck constant",
        "symbol": sp.Symbol("h"),
    },
    "hbar": {
        "value": 1.054571817e-34,
        "unit": "J·s",
        "name": "Reduced Planck constant",
        "symbol": sp.Symbol("ℏ"),
    },
    "G": {
        "value": 6.67430e-11,
        "unit": "m³/(kg·s²)",
        "name": "Gravitational constant",
        "symbol": sp.Symbol("G"),
    },
    "e": {
        "value": 1.602176634e-19,
        "unit": "C",
        "name": "Elementary charge",
        "symbol": sp.Symbol("e"),
    },
    "k_B": {
        "value": 1.380649e-23,
        "unit": "J/K",
        "name": "Boltzmann constant",
        "symbol": sp.Symbol("k_B"),
    },
    "N_A": {
        "value": 6.02214076e23,
        "unit": "1/mol",
        "name": "Avogadro constant",
        "symbol": sp.Symbol("N_A"),
    },
    # Electromagnetic constants
    "epsilon_0": {
        "value": 8.8541878128e-12,
        "unit": "F/m",
        "name": "Vacuum permittivity",
        "symbol": sp.Symbol("ε_0"),
    },
    "mu_0": {
        "value": 1.25663706212e-6,
        "unit": "H/m",
        "name": "Vacuum permeability",
        "symbol": sp.Symbol("μ_0"),
    },
    # Atomic/particle constants
    "m_e": {
        "value": 9.1093837015e-31,
        "unit": "kg",
        "name": "Electron mass",
        "symbol": sp.Symbol("m_e"),
    },
    "m_p": {
        "value": 1.67262192369e-27,
        "unit": "kg",
        "name": "Proton mass",
        "symbol": sp.Symbol("m_p"),
    },
    "m_n": {
        "value": 1.67492749804e-27,
        "unit": "kg",
        "name": "Neutron mass",
        "symbol": sp.Symbol("m_n"),
    },
    # Wien's displacement law constant
    "b_wien": {
        "value": 2.897771955e-3,
        "unit": "m·K",
        "name": "Wien's displacement constant",
        "symbol": sp.Symbol("b"),
    },
    # Stefan-Boltzmann constant
    "sigma": {
        "value": 5.670374419e-8,
        "unit": "W/(m²·K⁴)",
        "name": "Stefan-Boltzmann constant",
        "symbol": sp.Symbol("σ"),
    },
    # Gas constant
    "R": {
        "value": 8.314462618,
        "unit": "J/(mol·K)",
        "name": "Gas constant",
        "symbol": sp.Symbol("R"),
    },
    # Standard values
    "g": {
        "value": 9.80665,
        "unit": "m/s²",
        "name": "Standard gravity",
        "symbol": sp.Symbol("g"),
    },
    "atm": {
        "value": 101325,
        "unit": "Pa",
        "name": "Standard atmosphere",
        "symbol": sp.Symbol("atm"),
    },
}


def get_constant_value(name: str) -> float:
    """Get the numerical value of a constant."""
    if name in PHYSICAL_CONSTANTS:
        return PHYSICAL_CONSTANTS[name]["value"]
    raise KeyError(f"Unknown constant: {name}")


def get_constant_symbol(name: str) -> sp.Symbol:
    """Get the SymPy symbol for a constant."""
    if name in PHYSICAL_CONSTANTS:
        return PHYSICAL_CONSTANTS[name]["symbol"]
    raise KeyError(f"Unknown constant: {name}")


def get_substitution_dict() -> dict:
    """
    Get a dictionary mapping symbols to values for numerical substitution.

    Usage:
        expr.subs(get_substitution_dict())
    """
    return {info["symbol"]: info["value"] for info in PHYSICAL_CONSTANTS.values()}


def list_constants() -> list:
    """List all available constants with their descriptions."""
    return [
        {
            "name": name,
            "description": info["name"],
            "value": info["value"],
            "unit": info["unit"],
        }
        for name, info in PHYSICAL_CONSTANTS.items()
    ]
