import polars as pl
import pytest
from polars_units import UExpr


def test_uexpr_all_computational_methods():
    df = pl.DataFrame(
        {
            "distance": [1.0, 2.0, 3.0, 4.0],
            "group": ["a", "a", "b", "b"],
            "offset": [0.1, 0.2, 0.3, 0.4],
        }
    )
    dist = UExpr.col("distance", "meter")
    offset = UExpr.col("offset", "meter")

    # mean, sum, windowed mean, sum, conversion, filtering, groupby aggregation
    # Test mean aggregation
    result = df.select((dist.mean().alias("mean_dist")))
    assert result["mean_dist"][0] == pytest.approx(2.5)

    # Test sum aggregation
    result = df.select((dist.sum().alias("sum_dist")))
    assert result["sum_dist"][0] == pytest.approx(10.0)

    # Test over (window function)
    result = df.with_columns(
        [
            dist.mean().over("group").alias("group_mean"),
            (dist + offset).mean().over("group").alias("group_mean_sum"),
        ]
    )
    assert result["group_mean"].to_list() == [1.5, 1.5, 3.5, 3.5]
    assert result["group_mean_sum"].to_list() == pytest.approx([1.65, 1.65, 3.85, 3.85])

    # Test using .to() for unit conversion inside DataFrame
    result = df.with_columns(
        [
            dist.to("cm").alias("distance_cm"),
        ]
    )
    assert result["distance_cm"].to_list() == pytest.approx([100, 200, 300, 400])

    # Test filtering with UExpr comparisons
    mask = dist > UExpr.col("offset", "meter")
    filtered = df.filter(mask)
    assert filtered.shape[0] == 4  # all distances > offsets

    # Test groupby aggregation with UExpr
    grouped = df.group_by("group").agg(
        [
            dist.mean().alias("mean_dist"),
            (dist + offset).mean().alias("mean_sum"),
        ]
    )
    # Compare sorted results to avoid order mismatch
    assert sorted(grouped["mean_dist"].to_list()) == pytest.approx(sorted([1.5, 3.5]))
    assert sorted(grouped["mean_sum"].to_list()) == pytest.approx(sorted([1.65, 3.85]))

    # min, max, abs, neg, sqrt, pow
    result = df.select(
        [
            dist.min().alias("min_dist"),
            dist.max().alias("max_dist"),
            abs(dist).alias("abs_dist"),
            (-dist).alias("neg_dist"),
            dist.sqrt().alias("sqrt_dist"),
            (dist**2).alias("pow2_dist"),
        ]
    )
    assert result["min_dist"][0] == 1.0
    assert result["max_dist"][0] == 4.0
    assert result["abs_dist"].to_list() == [1.0, 2.0, 3.0, 4.0]
    assert result["neg_dist"].to_list() == [-1.0, -2.0, -3.0, -4.0]
    assert result["sqrt_dist"].to_list() == pytest.approx(
        [1.0, 1.4142, 1.73205, 2.0], rel=1e-3
    )
    assert result["pow2_dist"].to_list() == pytest.approx([1.0, 4.0, 9.0, 16.0])

    # log, exp (dimensionless)
    dimless = UExpr.col("distance", "dimensionless")
    result = df.select(
        [
            dimless.log().alias("log_dist"),
            dimless.exp().alias("exp_dist"),
        ]
    )
    assert result["log_dist"].to_list() == pytest.approx(
        [0.0, 0.6931, 1.0986, 1.3863], rel=1e-3
    )
    assert result["exp_dist"].to_list() == pytest.approx(
        [2.7183, 7.3891, 20.0855, 54.5981], rel=1e-3
    )

    # dot, norm
    result = df.select(
        [
            dist.dot(offset).alias("dot_prod"),
            # Removed dist.norm() as Polars Expr has no norm method
        ]
    )
    assert result["dot_prod"][0] == pytest.approx(3.0)

    # windowed min, max, sum, abs, neg, sqrt, pow
    result = df.with_columns(
        [
            dist.min().over("group").alias("group_min"),
            dist.max().over("group").alias("group_max"),
            dist.sum().over("group").alias("group_sum"),
            abs(dist).over("group").alias("group_abs"),
            (-dist).over("group").alias("group_neg"),
            dist.sqrt().over("group").alias("group_sqrt"),
            (dist**2).over("group").alias("group_pow2"),
        ]
    )
    assert result["group_min"].to_list() == [1.0, 1.0, 3.0, 3.0]
    assert result["group_max"].to_list() == [2.0, 2.0, 4.0, 4.0]
    assert result["group_sum"].to_list() == [3.0, 3.0, 7.0, 7.0]
    assert result["group_abs"].to_list() == [1.0, 2.0, 3.0, 4.0]
    assert result["group_neg"].to_list() == [-1.0, -2.0, -3.0, -4.0]
    assert result["group_sqrt"].to_list() == pytest.approx(
        [1.0, 1.4142, 1.73205, 2.0], rel=1e-3
    )
    assert result["group_pow2"].to_list() == pytest.approx([1.0, 4.0, 9.0, 16.0])
