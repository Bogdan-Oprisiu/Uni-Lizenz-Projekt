import math

class UnitConverter:
    """
    Handles conversion between compatible physical units.
    Supports length (metric/imperial) and angle units.
    """

    def __init__(self):
        # Define base units for each category
        self.BASE_UNITS = {
            'length': 'cm',
            'angle': 'rad',
            # Add other categories like 'speed', 'acceleration' if needed
            'angular_acceleration': 'rad/s^2',
            'linear_acceleration': 'cm/s^2',
        }

        # Conversion factors TO the base unit
        self.CONVERSION_FACTORS = {
            # Length -> cm
            'cm': 1.0,
            'm': 100.0,
            'mm': 0.1,
            'in': 2.54,
            'inch': 2.54,
            'ft': 30.48,
            'feet': 30.48,
            'foot': 30.48,
            # Angle -> rad
            'rad': 1.0,
            'deg': math.pi / 180.0,
            'degree': math.pi / 180.0,
            # Angular Acceleration -> rad/s^2
            'rad/s^2': 1.0,
            'deg/s^2': math.pi / 180.0,
            # Linear Acceleration -> cm/s^2
            'cm/s^2': 1.0,
            'm/s^2': 100.0,
            # Add more units and categories as needed
        }

        # Map units to their category
        self.UNIT_CATEGORIES = {
            'cm': 'length', 'm': 'length', 'mm': 'length', 'in': 'length', 'inch': 'length', 'ft': 'length', 'feet': 'length', 'foot': 'length',
            'rad': 'angle', 'deg': 'angle', 'degree': 'angle',
            'rad/s^2': 'angular_acceleration', 'deg/s^2': 'angular_acceleration',
            'cm/s^2': 'linear_acceleration', 'm/s^2': 'linear_acceleration',
        }

    def _normalize_unit(self, unit):
        """Converts unit to lowercase for consistent lookup."""
        return unit.lower() if isinstance(unit, str) else None

    def get_category(self, unit):
        """Gets the category of a given unit."""
        norm_unit = self._normalize_unit(unit)
        return self.UNIT_CATEGORIES.get(norm_unit)

    def convert(self, value, from_unit, to_unit):
        """
        Converts a value from one unit to another compatible unit.

        Args:
            value (float): The numerical value to convert.
            from_unit (str): The unit of the input value.
            to_unit (str): The desired unit for the output value.

        Returns:
            tuple: A tuple containing:
                   - float: The converted value, or None if conversion fails.
                   - str: An error code ('UNIT_MISMATCH', 'UNKNOWN_UNIT'),
                          or None if conversion succeeds.
        """
        norm_from = self._normalize_unit(from_unit)
        norm_to = self._normalize_unit(to_unit)

        if not norm_from or not norm_to:
            return None, "UNKNOWN_UNIT" # Or handle more gracefully

        if norm_from == norm_to:
            return float(value), None # No conversion needed

        from_category = self.get_category(norm_from)
        to_category = self.get_category(norm_to)

        # Check if units are known and compatible
        if not from_category or not to_category:
            return None, "UNKNOWN_UNIT"
        if from_category != to_category:
            return None, "UNIT_MISMATCH" # Cannot convert between different categories

        # Get conversion factors to the base unit for the category
        factor_from = self.CONVERSION_FACTORS.get(norm_from)
        factor_to = self.CONVERSION_FACTORS.get(norm_to)

        if factor_from is None or factor_to is None:
             # This shouldn't happen if UNIT_CATEGORIES is consistent with CONVERSION_FACTORS
             print(f"Warning: Missing conversion factor for known unit: {norm_from} or {norm_to}")
             return None, "INTERNAL_ERROR"

        # Convert: value -> base_unit -> to_unit
        try:
            value_in_base = float(value) * factor_from
            converted_value = value_in_base / factor_to
            return converted_value, None
        except (ValueError, TypeError):
             return None, "INVALID_PARAMETER_TYPE" # Input value wasn't a number
        except ZeroDivisionError:
             # Should not happen if factors are defined correctly
             print(f"Warning: Zero division error during conversion from {norm_from} to {norm_to}")
             return None, "INTERNAL_ERROR"


# =============================================================================
# Example Usage (for UnitConverter)
# =============================================================================
if __name__ == "__main__":
    converter = UnitConverter()

    print("--- UnitConverter Tests ---")

    # Length
    val_m = 1.5
    val_cm, err = converter.convert(val_m, "m", "cm")
    print(f"{val_m} m -> cm: {val_cm} (Error: {err})") # Expected: 150.0

    val_in = 10
    val_cm_in, err = converter.convert(val_in, "inch", "cm")
    print(f"{val_in} inch -> cm: {val_cm_in} (Error: {err})") # Expected: 25.4

    val_ft = 2
    val_m_ft, err = converter.convert(val_ft, "ft", "m")
    print(f"{val_ft} ft -> m: {val_m_ft} (Error: {err})") # Expected: 0.6096

    # Angle
    val_deg = 90
    val_rad, err = converter.convert(val_deg, "DEG", "rad") # Test case insensitivity
    print(f"{val_deg} deg -> rad: {val_rad} (Error: {err})") # Expected: 1.570...

    val_rad_2 = math.pi
    val_deg_2, err = converter.convert(val_rad_2, "rad", "degree")
    print(f"{val_rad_2} rad -> deg: {val_deg_2} (Error: {err})") # Expected: 180.0

    # Mismatch
    val_mismatch, err = converter.convert(100, "cm", "rad")
    print(f"100 cm -> rad: {val_mismatch} (Error: {err})") # Expected: None, UNIT_MISMATCH

    # Unknown
    val_unknown, err = converter.convert(10, "furlong", "cm")
    print(f"10 furlong -> cm: {val_unknown} (Error: {err})") # Expected: None, UNKNOWN_UNIT

