"""Regression tests for :mod:`rcopula.dependence`: random tie-breaking and the
vectorised Kendall's tau matrix."""

from __future__ import annotations

import time

import numpy as np
import pytest
from scipy import stats

import rcopula as rc
from rcopula.dependence import _kendall_by_sign_products, cor_kendall, pseudo_obs


def _scipy_matrix(x: np.ndarray) -> np.ndarray:
    d = x.shape[1]
    out = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            out[i, j] = out[j, i] = stats.kendalltau(x[:, i], x[:, j]).statistic
    return out


class TestRandomTies:
    def test_large_values_are_still_broken_at_random(self) -> None:
        """A 1e-12 jitter cannot move 1e6; the ties came out in row order."""
        column = np.full(200, 1.0e6)
        first = [
            int(np.argmin(pseudo_obs(column, ties_method="random", random_state=seed)[:, 0]))
            for seed in range(40)
        ]
        # In order-of-appearance tie breaking, row 0 always gets rank 1.
        assert len(set(first)) > 20

    def test_random_ranks_are_a_permutation_respecting_distinct_values(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.integers(0, 4, size=(500, 2)).astype(float) * 1e8
        u = np.asarray(pseudo_obs(x, ties_method="random", random_state=1))
        for j in range(2):
            ranks = np.rint(u[:, j] * 501).astype(int)
            assert sorted(ranks.tolist()) == list(range(1, 501))
            # Distinct values keep their order.
            order = np.argsort(ranks)
            assert np.all(np.diff(x[order, j]) >= 0)

    def test_each_tied_row_is_equally_likely_to_rank_first(self) -> None:
        column = np.array([5.0, 5.0, 5.0, 1.0])
        counts = np.zeros(3)
        for seed in range(3000):
            u = pseudo_obs(column, ties_method="random", random_state=seed)[:, 0]
            counts[int(np.argmin(u[:3]))] += 1
        assert stats.chisquare(counts).pvalue > 1e-3

    def test_seeded_and_untied_results_are_reproducible(self) -> None:
        x = np.random.default_rng(3).normal(size=(30, 2))
        a = pseudo_obs(x, ties_method="random", random_state=7)
        np.testing.assert_array_equal(a, pseudo_obs(x, ties_method="random", random_state=7))
        np.testing.assert_array_equal(a, pseudo_obs(x))


class TestKendallMatrix:
    @pytest.mark.parametrize("n, d", [(2, 3), (25, 6), (120, 9), (301, 4)])
    def test_matches_scipy_exactly_with_ties(self, n: int, d: int) -> None:
        rng = np.random.default_rng(n * d)
        x = rng.normal(size=(n, d))
        x[:, 0] = rng.integers(0, 3, size=n)  # heavy ties
        x[:, 1] = np.round(x[:, 1], 1)  # light ties
        if d > 3:
            x[:, 3] = 7.0  # constant -> NaN
        if d > 4:
            x[: n // 2, 4] = np.inf
        assert np.array_equal(_kendall_by_sign_products(x), _scipy_matrix(x), equal_nan=True)

    def test_nan_column_gives_nan_like_scipy(self) -> None:
        x = np.random.default_rng(1).normal(size=(40, 3))
        x[5, 2] = np.nan
        assert np.array_equal(_kendall_by_sign_products(x), _scipy_matrix(x), equal_nan=True)

    def test_public_function_matches_scipy_on_copula_data(self) -> None:
        u = rc.ClaytonCopula(2.0, dim=12).rvs(150, random_state=0)
        u = np.round(u, 2)
        assert np.array_equal(cor_kendall(u), _scipy_matrix(u), equal_nan=True)

    def test_dataframe_input(self) -> None:
        import pandas as pd

        x = np.random.default_rng(2).normal(size=(60, 4))
        np.testing.assert_array_equal(cor_kendall(pd.DataFrame(x)), _scipy_matrix(x))

    def test_many_columns_are_fast(self) -> None:
        """Pairwise scipy calls took minutes here; the blocked route takes seconds."""
        x = np.random.default_rng(0).normal(size=(500, 400))
        start = time.perf_counter()
        m = cor_kendall(x)
        assert time.perf_counter() - start < 30.0
        assert m[0, 1] == stats.kendalltau(x[:, 0], x[:, 1]).statistic
        assert m[398, 399] == stats.kendalltau(x[:, 398], x[:, 399]).statistic

    def test_rejects_one_dimensional_input(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            cor_kendall(np.arange(5.0))
