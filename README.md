# polars_units

Unit-aware expressions for Polars using Pint.

## Installation

```sh
pip install .
```

## Usage

```python
import polars as pl
from polars_units import UExpr

# Example: create a unit-aware column
height = UExpr.col("height", "meter")

# Use in Polars expressions
expr = (height + UExpr.col("offset", "meter")).to("cm")
```

See the source for full API details.
