import polars as pl
import pytest
from polars_units import UExpr

def test_col_and_unit():
    expr = UExpr.col("distance", "meter")
    assert str(expr.unit) == "meter"
    assert isinstance(expr.expr, pl.Expr)

def test_addition_same_unit():
    a = UExpr.col("a", "meter")
    b = UExpr.col("b", "meter")
    c = a + b
    assert str(c.unit) == "meter"

def test_addition_different_unit():
    a = UExpr.col("a", "meter")
    b = UExpr.col("b", "centimeter")
    c = a + b
    assert str(c.unit) == "meter"

def test_multiplication():
    a = UExpr.col("a", "meter")
    b = UExpr.col("b", "second")
    c = a * b
    assert str(c.unit) == "meter * second"

    d = a * 5
    assert str(d.unit) == "meter"

    e = 2 * a
    assert str(e.unit) == "meter"

def test_division():
    a = UExpr.col("a", "meter")
    b = UExpr.col("b", "second")
    c = a / b
    assert str(c.unit) == "meter / second"

    d = a / 2
    assert str(d.unit) == "meter"

    e = 10 / a
    assert str(e.unit) == "1 / meter"

def test_power_and_sqrt():
    a = UExpr.col("a", "meter")
    b = a ** 2
    assert str(b.unit) == "meter ** 2"
    c = a.sqrt()
    assert str(c.unit) == "meter ** 0.5"

def test_comparisons():
    a = UExpr.col("a", "meter")
    b = UExpr.col("b", "meter")
    expr = a < b
    assert isinstance(expr, pl.Expr)

    expr = a == b
    assert isinstance(expr, pl.Expr)

    with pytest.raises(TypeError):
        a == 5

    with pytest.raises(TypeError):
        a != 5

def test_dimensionless_functions():
    a = UExpr.col("x", "dimensionless")
    log_expr = a.log()
    exp_expr = a.exp()
    assert str(log_expr.unit) == "dimensionless"
    assert str(exp_expr.unit) == "dimensionless"

    b = UExpr.col("y", "meter")
    with pytest.raises(Exception):
        b.log()
    with pytest.raises(Exception):
        b.exp()

def test_aggregations():
    a = UExpr.col("a", "meter")
    assert str(a.sum().unit) == "meter"
    assert str(a.mean().unit) == "meter"
    assert str(a.min().unit) == "meter"
    assert str(a.max().unit) == "meter"

def test_alias_repr():
    a = UExpr.col("a", "meter")
    expr = a.alias("distance")
    assert isinstance(expr, pl.Expr)
    assert "UExpr" in repr(a)
