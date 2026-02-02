"""
Unit handling with Pint integration.

Provides unit parsing, conversion, and dimensionality checking
for physics calculations in MathSolver.
"""

from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
import re

# Try to import Pint, provide fallback if not installed
try:
    import pint

    PINT_AVAILABLE = True
except ImportError:
    PINT_AVAILABLE = False
    pint = None


@dataclass
class UnitValue:
    """
    A numerical value with associated units.

    Wraps Pint Quantity for use in MathSolver.
    """

    magnitude: float
    unit_str: str
    _quantity: Any = None  # pint.Quantity if available

    @property
    def quantity(self) -> Any:
        """Get the Pint Quantity object."""
        return self._quantity

    def to(self, target_unit: str) -> "UnitValue":
        """Convert to different units."""
        if self._quantity is not None:
            converted = self._quantity.to(target_unit)
            return UnitValue(
                magnitude=converted.magnitude,
                unit_str=str(converted.units),
                _quantity=converted,
            )
        return self  # No conversion possible without Pint

    def to_base_units(self) -> "UnitValue":
        """Convert to SI base units."""
        if self._quantity is not None:
            converted = self._quantity.to_base_units()
            return UnitValue(
                magnitude=converted.magnitude,
                unit_str=str(converted.units),
                _quantity=converted,
            )
        return self

    @property
    def dimensionality(self) -> str:
        """Get dimensionality string (e.g., '[length]', '[mass] * [time] ** -2')."""
        if self._quantity is not None:
            return str(self._quantity.dimensionality)
        return "unknown"

    def __str__(self) -> str:
        return f"{self.magnitude} {self.unit_str}"

    def __repr__(self) -> str:
        return f"UnitValue({self.magnitude}, '{self.unit_str}')"


class UnitHandler:
    """
    Handles unit parsing, conversion, and validation.

    Uses Pint library for unit operations. Falls back to basic
    string parsing if Pint is not available.

    Usage:
        handler = UnitHandler()
        value = handler.parse("5.5 m/s")
        converted = handler.convert(value, "km/h")
        if handler.check_dimensionality(val1, val2):
            result = val1.magnitude + val2.to(val1.unit_str).magnitude
    """

    # Common physics unit aliases
    UNIT_ALIASES = {
        "angstrom": "angstrom",
        "Å": "angstrom",
        "A": "angstrom",  # Context-dependent, could be Ampere
        "eV": "electron_volt",
        "keV": "kiloelectron_volt",
        "MeV": "megaelectron_volt",
        "GeV": "gigaelectron_volt",
        "amu": "atomic_mass_unit",
        "u": "atomic_mass_unit",
        "c": "speed_of_light",
    }

    def __init__(self):
        """Initialize the unit handler."""
        self._ureg = None
        if PINT_AVAILABLE:
            self._ureg = pint.UnitRegistry()
            # Add custom definitions for physics
            self._add_physics_units()

    def _add_physics_units(self):
        """Add physics-specific unit definitions."""
        if self._ureg is None:
            return

        # Natural units and particle physics
        try:
            # Speed of light (already in Pint as 'c')
            self._ureg.define("natural_length = hbar * c / eV")
        except pint.errors.RedefinitionError:
            pass

    @property
    def available(self) -> bool:
        """Check if Pint is available for full unit support."""
        return PINT_AVAILABLE and self._ureg is not None

    def parse(self, value_str: str) -> Optional[UnitValue]:
        """
        Parse a string with value and units.

        Args:
            value_str: String like "5.5 m/s", "3e8 m/s", "1.6e-19 C"

        Returns:
            UnitValue or None if parsing fails
        """
        if not value_str or not value_str.strip():
            return None

        value_str = value_str.strip()

        # Try to extract number and unit
        match = re.match(
            r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*)$", value_str
        )

        if not match:
            return None

        magnitude_str, unit_str = match.groups()

        try:
            magnitude = float(magnitude_str)
        except ValueError:
            return None

        unit_str = unit_str.strip() or "dimensionless"

        # Apply aliases
        unit_str = self.UNIT_ALIASES.get(unit_str, unit_str)

        if self._ureg is not None:
            try:
                quantity = self._ureg.Quantity(magnitude, unit_str)
                return UnitValue(
                    magnitude=magnitude,
                    unit_str=str(quantity.units),
                    _quantity=quantity,
                )
            except Exception:
                # Fall through to basic parsing
                pass

        # Basic fallback (no Pint)
        return UnitValue(magnitude=magnitude, unit_str=unit_str)

    def create(self, magnitude: float, unit_str: str) -> UnitValue:
        """
        Create a UnitValue from magnitude and unit string.

        Args:
            magnitude: Numerical value
            unit_str: Unit string (e.g., "m/s", "J", "kg*m**2/s**2")

        Returns:
            UnitValue
        """
        unit_str = self.UNIT_ALIASES.get(unit_str, unit_str)

        if self._ureg is not None:
            try:
                quantity = self._ureg.Quantity(magnitude, unit_str)
                return UnitValue(
                    magnitude=magnitude,
                    unit_str=str(quantity.units),
                    _quantity=quantity,
                )
            except Exception:
                pass

        return UnitValue(magnitude=magnitude, unit_str=unit_str)

    def convert(self, value: UnitValue, target_unit: str) -> Optional[UnitValue]:
        """
        Convert a value to different units.

        Args:
            value: Source UnitValue
            target_unit: Target unit string

        Returns:
            Converted UnitValue or None if conversion fails
        """
        if value._quantity is None:
            return None

        target_unit = self.UNIT_ALIASES.get(target_unit, target_unit)

        try:
            converted = value._quantity.to(target_unit)
            return UnitValue(
                magnitude=converted.magnitude,
                unit_str=str(converted.units),
                _quantity=converted,
            )
        except Exception:
            return None

    def to_si(self, value: UnitValue) -> Optional[UnitValue]:
        """
        Convert value to SI base units.

        Args:
            value: UnitValue to convert

        Returns:
            UnitValue in SI base units
        """
        if value._quantity is None:
            return value

        try:
            converted = value._quantity.to_base_units()
            return UnitValue(
                magnitude=converted.magnitude,
                unit_str=str(converted.units),
                _quantity=converted,
            )
        except Exception:
            return value

    def check_dimensionality(self, value1: UnitValue, value2: UnitValue) -> bool:
        """
        Check if two values have compatible dimensions.

        Args:
            value1: First UnitValue
            value2: Second UnitValue

        Returns:
            True if dimensionally compatible (can be added/compared)
        """
        if value1._quantity is None or value2._quantity is None:
            return False

        return value1._quantity.dimensionality == value2._quantity.dimensionality

    def get_dimensionality(self, unit_str: str) -> Optional[str]:
        """
        Get the dimensionality of a unit string.

        Args:
            unit_str: Unit string (e.g., "m/s", "J")

        Returns:
            Dimensionality string or None
        """
        if self._ureg is None:
            return None

        unit_str = self.UNIT_ALIASES.get(unit_str, unit_str)

        try:
            unit = self._ureg.Unit(unit_str)
            return str(unit.dimensionality)
        except Exception:
            return None

    def format_with_unit(
        self,
        magnitude: float,
        unit_str: str,
        precision: int = 4,
        scientific: bool = True,
    ) -> str:
        """
        Format a value with units for display.

        Args:
            magnitude: Numerical value
            unit_str: Unit string
            precision: Number of significant figures
            scientific: Use scientific notation for large/small numbers

        Returns:
            Formatted string
        """
        if scientific and (
            abs(magnitude) > 1e4 or (abs(magnitude) < 1e-3 and magnitude != 0)
        ):
            return f"{magnitude:.{precision}e} {unit_str}"
        else:
            return f"{magnitude:.{precision}g} {unit_str}"

    def validate_physics_units(
        self, variable_name: str, value: UnitValue, expected_unit: str
    ) -> Tuple[bool, str]:
        """
        Validate that a value has the expected physical units.

        Args:
            variable_name: Name of the variable (for error message)
            value: UnitValue to check
            expected_unit: Expected unit string

        Returns:
            (is_valid, error_message)
        """
        if value._quantity is None or self._ureg is None:
            return True, ""  # Can't validate without Pint

        expected_unit = self.UNIT_ALIASES.get(expected_unit, expected_unit)

        try:
            expected = self._ureg.Unit(expected_unit)

            if value._quantity.dimensionality != expected.dimensionality:
                return False, (
                    f"{variable_name} has dimensionality {value.dimensionality}, "
                    f"expected {expected.dimensionality}"
                )

            return True, ""

        except Exception as e:
            return False, f"Could not validate units: {e}"


# Module-level convenience functions

_handler: Optional[UnitHandler] = None


def get_handler() -> UnitHandler:
    """Get the global unit handler instance."""
    global _handler
    if _handler is None:
        _handler = UnitHandler()
    return _handler


def parse_value(value_str: str) -> Optional[UnitValue]:
    """Parse a value string with units."""
    return get_handler().parse(value_str)


def convert_units(magnitude: float, from_unit: str, to_unit: str) -> Optional[float]:
    """
    Convert a magnitude from one unit to another.

    Args:
        magnitude: Numerical value
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted magnitude or None if conversion fails
    """
    handler = get_handler()
    value = handler.create(magnitude, from_unit)
    converted = handler.convert(value, to_unit)
    return converted.magnitude if converted else None


def are_compatible(unit1: str, unit2: str) -> bool:
    """Check if two unit strings are dimensionally compatible."""
    handler = get_handler()
    val1 = handler.create(1.0, unit1)
    val2 = handler.create(1.0, unit2)
    return handler.check_dimensionality(val1, val2)
