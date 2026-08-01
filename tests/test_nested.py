"""Tests for nested Archimedean copulas and the tilted-stable sampler.

Nesting is the one construction where the CDF and the sampler are built from
genuinely different mathematics -- one composes generators, the other runs a
hierarchy of frailties -- so the check that matters is that they agree. Every
tree below is validated by simulating from it and comparing the empirical joint
distribution against the analytic CDF at random points.

The tilted stable underneath is checked against its defining Laplace transform,
which is exact.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from scipy import stats

import rcopula as rc
from rcopula.special.stable import retstable
from rcopula.structural import NestedArchimedean, fit_nested


def two_block(family: type, outer: float, first: float, second: float) -> NestedArchimedean:
    return NestedArchimedean(
        family(outer),
        children=[
            NestedArchimedean(family(first), [0, 1, 2]),
            NestedArchimedean(family(second), [3, 4]),
        ],
    )


def three_deep(family: type, a: float, b: float, c: float) -> NestedArchimedean:
    return NestedArchimedean(
        family(a),
        [4],
        [NestedArchimedean(family(b), [3], [NestedArchimedean(family(c), [0, 1, 2])])],
    )


TREES = [
    ("gumbel two-block", two_block(rc.GumbelCopula, 1.5, 4.0, 3.0)),
    ("gumbel three-deep", three_deep(rc.GumbelCopula, 1.3, 2.0, 5.0)),
    ("clayton two-block", two_block(rc.ClaytonCopula, 0.5, 3.0, 2.0)),
    ("clayton three-deep", three_deep(rc.ClaytonCopula, 0.4, 1.5, 4.0)),
    ("gumbel flat", NestedArchimedean(rc.GumbelCopula(2.5), [0, 1, 2, 3, 4])),
    ("clayton flat", NestedArchimedean(rc.ClaytonCopula(2.0), [0, 1, 2])),
]
IDS = [name for name, _ in TREES]
COPULAS = [tree for _, tree in TREES]


class TestTiltedStable:
    """``retstable`` is the reason nested Clayton has never existed in Python."""

    @pytest.mark.parametrize("alpha", [0.2, 0.5, 0.8, 0.95])
    @pytest.mark.parametrize(("v0", "h"), [(1.0, 1.0), (0.3, 2.0), (5.0, 1.0), (0.05, 1.0)])
    def test_it_matches_its_defining_laplace_transform(
        self, alpha: float, v0: float, h: float
    ) -> None:
        r"""``E[e^{-tV}] = exp(-v0 [(t+h)^alpha - h^alpha])``, exactly."""
        draws = retstable(200_000, alpha, v0, h, np.random.default_rng(0))
        for t in (0.3, 1.0, 4.0):
            want = np.exp(-v0 * ((t + h) ** alpha - h**alpha))
            assert np.mean(np.exp(-t * draws)) == pytest.approx(want, abs=3e-3)

    @pytest.mark.parametrize("alpha", [0.3, 0.6, 0.9])
    @pytest.mark.parametrize("v0", [0.5, 2.0, 20.0])
    def test_the_mean_follows_from_the_transform(self, alpha: float, v0: float) -> None:
        """Differentiating at zero gives ``alpha v0 h^(alpha-1)``."""
        h = 1.5
        draws = retstable(200_000, alpha, v0, h, np.random.default_rng(0))
        assert draws.mean() == pytest.approx(alpha * v0 * h ** (alpha - 1.0), rel=0.03)

    def test_a_zero_tilt_gives_the_untilted_stable(self) -> None:
        draws = retstable(200_000, 0.5, 1.0, 0.0, np.random.default_rng(0))
        for t in (0.5, 2.0):
            assert np.mean(np.exp(-t * draws)) == pytest.approx(np.exp(-(t**0.5)), abs=3e-3)

    def test_alpha_one_is_degenerate(self) -> None:
        assert np.all(retstable(50, 1.0, 3.0, 1.0, np.random.default_rng(0)) == 3.0)

    def test_v0_may_vary_per_draw(self) -> None:
        """Which is what the nested sampler needs: one outer frailty per row."""
        v0 = np.array([0.1, 1.0, 10.0])
        draws = retstable(3, 0.5, v0, 1.0, np.random.default_rng(0))
        assert draws.shape == (3,)
        assert np.all(draws > 0)

    def test_the_cost_is_linear_rather_than_exponential_in_the_tilt(self) -> None:
        """Naive rejection accepts with probability ``exp(-v0 h^alpha)``.

        Splitting ``v0`` by infinite divisibility puts every piece's acceptance
        at about ``e^-1``, so a hundredfold larger ``v0`` costs about a hundred
        times more rather than ``e^100`` times more. Without that, the expected
        number of attempts is *infinite* whenever ``v0 ~ Gamma(k, 1)`` with
        ``k >= 1``, which is exactly the nested Clayton case.
        """
        import time

        rng = np.random.default_rng(0)
        timings = []
        for v0 in (1.0, 100.0):
            start = time.perf_counter()
            retstable(20_000, 0.5, v0, 1.0, rng)
            timings.append(time.perf_counter() - start)
        assert timings[1] < 400 * timings[0]

    def test_it_rejects_bad_parameters(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="alpha must lie"):
            retstable(10, 1.5, 1.0, 1.0, rng)
        with pytest.raises(ValueError, match="h must be non-negative"):
            retstable(10, 0.5, 1.0, -1.0, rng)
        with pytest.raises(ValueError, match="v0 must be non-negative"):
            retstable(10, 0.5, -1.0, 1.0, rng)


class TestTheSamplerAgreesWithTheCdf:
    """The central check: two independent constructions of the same object."""

    @pytest.mark.parametrize("cop", COPULAS, ids=IDS)
    def test_the_empirical_joint_matches_the_analytic_cdf(self, cop: NestedArchimedean) -> None:
        sample = cop.rvs(150_000, random_state=0)
        points = np.random.default_rng(1).uniform(0.15, 0.9, size=(12, cop.dim))
        empirical = np.array([np.mean(np.all(sample <= p, axis=1)) for p in points])
        assert np.allclose(cop.cdf(points), empirical, atol=6e-3)

    @pytest.mark.parametrize("cop", COPULAS, ids=IDS)
    def test_margins_are_uniform(self, cop: NestedArchimedean) -> None:
        sample = cop.rvs(60_000, random_state=0)
        for j in range(cop.dim):
            assert stats.kstest(sample[:, j], "uniform").pvalue > 1e-3

    @pytest.mark.parametrize("cop", COPULAS, ids=IDS)
    def test_the_cdf_has_uniform_margins(self, cop: NestedArchimedean) -> None:
        grid = np.linspace(0.05, 0.95, 15)
        for j in range(cop.dim):
            points = np.ones((grid.size, cop.dim))
            points[:, j] = grid
            assert np.allclose(cop.cdf(points), grid, atol=1e-12)

    @pytest.mark.parametrize("cop", COPULAS, ids=IDS)
    def test_draws_stay_strictly_inside_the_cube(self, cop: NestedArchimedean) -> None:
        sample = cop.rvs(40_000, random_state=0)
        assert np.all((sample > 0.0) & (sample < 1.0))


class TestDependenceVariesByBranch:
    """The reason to nest at all."""

    @pytest.mark.parametrize("cop", COPULAS, ids=IDS)
    def test_analytic_tau_matches_the_sample(self, cop: NestedArchimedean) -> None:
        sample = cop.rvs(120_000, random_state=0)
        assert np.allclose(cop.tau_matrix(), rc.cor_kendall(sample), atol=6e-3)

    def test_a_tree_produces_taus_no_flat_copula_can(self) -> None:
        cop = two_block(rc.GumbelCopula, 1.5, 4.0, 3.0)
        tau = cop.tau_matrix()
        assert tau[0, 1] == pytest.approx(0.75)  # inside the first block
        assert tau[3, 4] == pytest.approx(2 / 3)  # inside the second
        assert tau[0, 3] == pytest.approx(1 / 3)  # only at the root
        assert len({round(tau[0, 1], 6), round(tau[3, 4], 6), round(tau[0, 3], 6)}) == 3

    def test_dependence_is_governed_by_the_lowest_common_ancestor(self) -> None:
        cop = three_deep(rc.GumbelCopula, 1.3, 2.0, 5.0)
        assert cop.lowest_common_ancestor(0, 1).theta == 5.0  # deepest block
        assert cop.lowest_common_ancestor(0, 3).theta == 2.0  # one level up
        assert cop.lowest_common_ancestor(0, 4).theta == 1.3  # the root

    def test_tail_dependence_varies_by_branch_too(self) -> None:
        cop = two_block(rc.GumbelCopula, 1.5, 4.0, 3.0)
        _, upper = cop.lambda_matrix()
        assert upper[0, 1] > upper[3, 4] > upper[0, 3]

    def test_a_single_scalar_measure_is_refused(self) -> None:
        """Returning one number would be answering a question the object denies."""
        cop = two_block(rc.GumbelCopula, 1.5, 4.0, 3.0)
        for method in (cop.tau, cop.rho, cop.lambda_):
            with pytest.raises(NotImplementedError, match="pair"):
                method()


class TestStructureIsValidated:
    def test_the_nesting_condition_is_enforced(self) -> None:
        """Dependence may increase as you descend, never decrease."""
        inner = NestedArchimedean(rc.GumbelCopula(1.5), [0, 1])
        with pytest.raises(ValueError, match="never decrease"):
            NestedArchimedean(rc.GumbelCopula(3.0), [2], [inner])

    def test_equal_parameters_are_allowed(self) -> None:
        """The boundary case: the child simply adds nothing."""
        inner = NestedArchimedean(rc.GumbelCopula(2.0), [0, 1])
        cop = NestedArchimedean(rc.GumbelCopula(2.0), [2], [inner])
        assert cop.tau_matrix()[0, 2] == pytest.approx(cop.tau_matrix()[0, 1])

    def test_mixing_families_is_refused(self) -> None:
        inner = NestedArchimedean(rc.ClaytonCopula(3.0), [0, 1])
        with pytest.raises(ValueError, match="mixing families"):
            NestedArchimedean(rc.GumbelCopula(1.5), [2], [inner])

    def test_repeated_variables_are_caught(self) -> None:
        inner = NestedArchimedean(rc.GumbelCopula(3.0), [0, 1])
        with pytest.raises(ValueError, match="variables repeat"):
            NestedArchimedean(rc.GumbelCopula(1.5), [1], [inner])

    def test_a_sub_tree_is_not_a_copula(self) -> None:
        """It legitimately covers [3, 4]; only a root covers 0..d-1."""
        branch = NestedArchimedean(rc.GumbelCopula(3.0), [3, 4])
        with pytest.raises(ValueError, match="sub-tree rather than a copula"):
            branch.cdf([[0.5, 0.5]])

    def test_the_generator_must_be_archimedean(self) -> None:
        with pytest.raises(TypeError, match="one-parameter Archimedean"):
            NestedArchimedean(rc.GaussianCopula(0.5), [0, 1])

    def test_unsampleable_families_are_refused_rather_than_approximated(self) -> None:
        cop = NestedArchimedean(
            rc.FrankCopula(2.0), [2], [NestedArchimedean(rc.FrankCopula(5.0), [0, 1])]
        )
        with pytest.raises(NotImplementedError, match="conditional inner frailty"):
            cop.rvs(10, random_state=0)

    def test_the_density_is_refused_with_a_useful_message(self) -> None:
        cop = two_block(rc.GumbelCopula, 1.5, 4.0, 3.0)
        with pytest.raises(NotImplementedError, match="fit_nested"):
            cop.pdf(np.full((1, 5), 0.5))

    def test_structure_helpers(self) -> None:
        cop = three_deep(rc.GumbelCopula, 1.3, 2.0, 5.0)
        assert cop.dim == 5
        assert cop.depth == 3
        assert len(cop.nodes()) == 3
        assert sorted(cop.leaves()) == [0, 1, 2, 3, 4]
        assert "Gumbel(theta=1.3)" in cop.describe()
        assert "depth 3" in repr(cop)


class TestEstimation:
    """No density, so estimation inverts pairwise Kendall's tau."""

    TRUTHS: ClassVar[list] = [
        two_block(rc.GumbelCopula, 1.5, 4.0, 3.0),
        three_deep(rc.GumbelCopula, 1.3, 2.0, 5.0),
        two_block(rc.ClaytonCopula, 0.5, 3.0, 2.0),
        three_deep(rc.ClaytonCopula, 0.4, 1.5, 4.0),
    ]

    @pytest.mark.parametrize("truth", TRUTHS)
    def test_every_node_is_recovered(self, truth: NestedArchimedean) -> None:
        fitted = fit_nested(truth, truth.rvs(6000, random_state=0))
        for got, want in zip(fitted.nodes(), truth.nodes(), strict=True):
            assert got.theta == pytest.approx(want.theta, rel=0.10)

    def test_the_fitted_tree_keeps_its_shape(self) -> None:
        truth = three_deep(rc.GumbelCopula, 1.3, 2.0, 5.0)
        fitted = fit_nested(truth, truth.rvs(2000, random_state=0))
        assert fitted.depth == truth.depth
        assert fitted.leaves() == truth.leaves()

    def test_the_estimates_respect_the_nesting_condition(self) -> None:
        """Sampling noise can otherwise leave a child below its parent, which
        would not be a copula at all -- so the estimates are floored."""
        truth = two_block(rc.GumbelCopula, 2.0, 2.05, 2.05)  # nearly indistinguishable
        for seed in range(8):
            fitted = fit_nested(truth, truth.rvs(300, random_state=seed))
            for node in fitted.nodes():
                for child in node.children:
                    assert child.theta >= node.theta - 1e-12

    def test_it_only_uses_ranks(self) -> None:
        """Invariant under strictly INCREASING marginal transforms.

        Not under decreasing ones -- those reverse the sign of every tau the
        column takes part in, and the estimator correctly notices.
        """
        truth = two_block(rc.ClaytonCopula, 0.5, 3.0, 2.0)
        u = truth.rvs(3000, random_state=0)
        warped = np.column_stack(
            [
                stats.norm.ppf(u[:, 0]),
                np.exp(u[:, 1]),
                u[:, 2] ** 3,
                stats.expon.ppf(u[:, 3]),
                -np.log1p(-u[:, 4]),
            ]
        )
        a = [n.theta for n in fit_nested(truth, u).nodes()]
        b = [n.theta for n in fit_nested(truth, warped).nodes()]
        assert np.allclose(a, b)

    def test_it_checks_the_data_shape(self) -> None:
        truth = two_block(rc.GumbelCopula, 1.5, 4.0, 3.0)
        with pytest.raises(ValueError, match="dim=5"):
            fit_nested(truth, np.random.default_rng(0).uniform(size=(50, 3)))

    def test_with_thetas_rebuilds_the_tree(self) -> None:
        truth = three_deep(rc.GumbelCopula, 1.3, 2.0, 5.0)
        rebuilt = truth.with_thetas([1.4, 2.5, 6.0])
        assert [n.theta for n in rebuilt.nodes()] == [1.4, 2.5, 6.0]
        assert rebuilt.leaves() == truth.leaves()
        with pytest.raises(ValueError, match="parameters for"):
            truth.with_thetas([1.0, 2.0])
