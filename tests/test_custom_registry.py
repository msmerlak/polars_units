import polars as pl
import pint
from polars_units import UExpr


def test_custom_registry():
    # Create a custom registry with a new unit
    custom_ureg = pint.UnitRegistry()
    custom_ureg.define("furlong = 201.168 m")
    # Use custom registry in UExpr
    dist = UExpr.col("distance", "furlong", unit_registry=custom_ureg)
    assert str(dist.unit) == "furlong"
    # Convert to meters using custom registry
    dist_m = dist.to("meter")
    assert str(dist_m.unit) == "meter"
    # Check conversion factor
    assert abs((1 * dist.unit).to("meter").magnitude - 201.168) < 1e-6
    # Arithmetic with custom registry
    dist2 = UExpr.col("distance2", "furlong", unit_registry=custom_ureg)
    sum_expr = dist + dist2
    assert str(sum_expr.unit) == "furlong"
    # Ensure registry is preserved
    assert sum_expr.ureg is custom_ureg


if __name__ == "__main__":
    test_custom_registry()
    print("Custom registry test passed.")
