import polars as pl
import pint
from typing import Union, Any


default_ureg = pint.UnitRegistry()
Unit = default_ureg.Unit
Quantity = default_ureg.Quantity


class UExpr:
    def __init__(
        self,
        expr: pl.Expr,
        unit: Union[str, Any],
        unit_registry: pint.UnitRegistry = None,
    ):
        self.expr = expr
        self.ureg = unit_registry if unit_registry is not None else default_ureg
        self.unit = self.ureg.Unit(unit) if isinstance(unit, str) else unit

    def __getattr__(self, name):
        # Never forward internal or dunder methods
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        try:
            return super().__getattribute__(name)
        except AttributeError:
            pass
        # List of methods that require dimensionless quantities
        dimensionless_methods = {
            "exp",
            "log",
            "log10",
            "log1p",
            "sin",
            "cos",
            "tan",
            "sinh",
            "cosh",
            "tanh",
            "arcsin",
            "arccos",
            "arctan",
            "arcsinh",
            "arccosh",
            "arctanh",
            "cot",
            "degrees",
            "radians",
            "entropy",
        }
        if name in dimensionless_methods:
            self._require_dimensionless(name)
        attr = getattr(self.expr, name)
        if callable(attr):

            def method(*args, **kwargs):
                # Unwrap UExpr arguments to pl.Expr
                def unwrap(arg):
                    if isinstance(arg, UExpr):
                        return arg.expr
                    return arg

                new_args = tuple(unwrap(a) for a in args)
                new_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
                result = attr(*new_args, **new_kwargs)
                # If result is a pl.Expr, wrap in UExpr with same unit
                if isinstance(result, pl.Expr):
                    unit = (
                        self.unit
                        if name not in dimensionless_methods
                        else self.ureg.dimensionless
                    )
                    return UExpr(result, unit, unit_registry=self.ureg)
                return result

            return method
        else:
            return attr

    # -----------------------
    # Constructors & helpers
    # -----------------------

    @classmethod
    def col(
        cls, name: str, unit: Union[str, Any], unit_registry: pint.UnitRegistry = None
    ) -> "UExpr":
        """
        Wrap a Polars column as a unit-aware expression.
        """
        ureg = unit_registry if unit_registry is not None else default_ureg
        return cls(pl.col(name), ureg.Unit(unit), unit_registry=ureg)

    def to(self, new_unit: Union[str, Any]) -> "UExpr":
        """
        Convert this quantity to a new unit by inserting a scalar factor.
        """
        new_unit = self.ureg.Unit(new_unit) if isinstance(new_unit, str) else new_unit
        factor = (1 * self.unit).to(new_unit).magnitude  # scalar
        return UExpr(self.expr * factor, new_unit, unit_registry=self.ureg)

    @property
    def dimensionality(self):
        return self.unit.dimensionality

    @property
    def is_dimensionless(self) -> bool:
        return bool(self.unit.dimensionless)

    def _require_dimensionless(self, op_name: str):
        if not self.is_dimensionless:
            raise pint.DimensionalityError(
                self.unit,
                self.ureg.dimensionless,
                f"{op_name} requires a dimensionless quantity",
            )

    # -----------------------
    # Arithmetic
    # -----------------------

    def __add__(self, other: "UExpr") -> "UExpr":
        return self._binary_op_same_dim(other, lambda a, b: a + b)

    def __sub__(self, other: "UExpr") -> "UExpr":
        return self._binary_op_same_dim(other, lambda a, b: a - b)

    def __mul__(self, other) -> "UExpr":
        if isinstance(other, UExpr):
            new_unit = self.unit * other.unit
            return UExpr(self.expr * other.expr, new_unit)
        elif isinstance(other, (int, float)):
            return UExpr(self.expr * other, self.unit)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other) -> "UExpr":
        if isinstance(other, UExpr):
            new_unit = self.unit / other.unit
            return UExpr(self.expr / other.expr, new_unit)
        elif isinstance(other, (int, float)):
            return UExpr(self.expr / other, self.unit)
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            new_unit = self.ureg.Unit("dimensionless") / self.unit
            return UExpr(other / self.expr, new_unit, unit_registry=self.ureg)
        else:
            return NotImplemented

    def __pow__(self, other) -> "UExpr":
        if isinstance(other, (int, float)):
            # Raise unit to a power
            new_unit = self.unit**other
            return UExpr(self.expr**other, new_unit)
        else:
            return NotImplemented

    def __rpow__(self, other):
        if isinstance(other, (int, float)):
            return UExpr(other**self.expr, self.unit)
        else:
            return NotImplemented

    def sqrt(self) -> "UExpr":
        return UExpr(self.expr.sqrt(), self.unit**0.5, unit_registry=self.ureg)

    def unwrap(self) -> pl.Expr:
        """Return the underlying pl.Expr for DataFrame operations."""
        return self.expr

    def alias(self, name: str) -> pl.Expr:
        return self.expr.alias(name)

    def __lt__(self, other: "UExpr") -> pl.Expr:
        return self._cmp_same_dim(other, lambda a, b: a < b)

    def __le__(self, other: "UExpr") -> pl.Expr:
        return self._cmp_same_dim(other, lambda a, b: a <= b)

    def __gt__(self, other: "UExpr") -> pl.Expr:
        return self._cmp_same_dim(other, lambda a, b: a > b)

    def __ge__(self, other: "UExpr") -> pl.Expr:
        return self._cmp_same_dim(other, lambda a, b: a >= b)

    def __eq__(self, other: object) -> pl.Expr:  # type: ignore[override]
        if not isinstance(other, UExpr):
            raise TypeError("Equality comparison requires another UExpr")
        return self._cmp_same_dim(other, lambda a, b: a == b)

    def __ne__(self, other: object) -> pl.Expr:  # type: ignore[override]
        if not isinstance(other, UExpr):
            raise TypeError("Inequality comparison requires another UExpr")
        return self._cmp_same_dim(other, lambda a, b: a != b)

    def __abs__(self) -> "UExpr":
        return UExpr(abs(self.expr), self.unit, unit_registry=self.ureg)

    def __neg__(self) -> "UExpr":
        return UExpr(-self.expr, self.unit, unit_registry=self.ureg)

    def _binary_op_same_dim(self, other, op):
        # Support arithmetic between UExprs with compatible units
        if isinstance(other, UExpr):
            # Convert other's unit to self's unit if possible
            if self.unit != other.unit:
                other_converted = other.to(self.unit)
            else:
                other_converted = other
            result_expr = op(self.expr, other_converted.expr)
            return UExpr(result_expr, self.unit, unit_registry=self.ureg)
        elif isinstance(other, (int, float)):
            result_expr = op(self.expr, other)
            return UExpr(result_expr, self.unit, unit_registry=self.ureg)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for operation: '{type(self)}' and '{type(other)}'"
            )

    def _cmp_same_dim(self, other, op):
        # Support comparisons between UExprs with compatible units
        if isinstance(other, UExpr):
            if self.unit != other.unit:
                other_converted = other.to(self.unit)
            else:
                other_converted = other
            return op(self.expr, other_converted.expr)
        elif isinstance(other, (int, float)):
            return op(self.expr, other)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for comparison: '{type(self)}' and '{type(other)}'"
            )
