"""Tests for vine copulas.

A vine has three separate recursions -- the density, the sampler and the
Rosenblatt transform -- that must all describe the same object, so most of these
check one against another rather than against a stored number:

* an **all-Gaussian vine is exactly a Gaussian copula**, so its density can be
  compared against `GaussianCopula` as an identity rather than a tolerance. This
  is the sharpest test available, and it is what caught a partial-correlation
  bug invisible below four dimensions;
* the density must integrate to one over the unit cube;
* the Rosenblatt transform of a simulated sample must be independent uniforms,
  which ties the sampler and the density together;
* tree-1 pair-copulas govern the pairs they join, so the sample tau must match.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import rcopula as rc
from rcopula.vine import VineCopula, _partial, fit_vine


def gaussian_vine(sigma: np.ndarray, structure: str) -> VineCopula:
    """Build the vine whose pair parameters are ``sigma``'s partial correlations."""
    d = sigma.shape[0]
    blank = VineCopula(
        [[rc.GaussianCopula(0.0)] * (d - 1 - k) for k in range(d - 1)], structure=structure
    )
    trees = [
        [
            rc.GaussianCopula(float(_partial(sigma, *blank._edge_indices(k, i))))
            for i in range(d - 1 - k)
        ]
        for k in range(d - 1)
    ]
    return VineCopula(trees, structure=structure)


def random_correlation(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(d, d + 4))
    s = a @ a.T
    scale = np.sqrt(np.diag(s))
    return s / np.outer(scale, scale)


MIXED = {
    "D": VineCopula(
        [[rc.ClaytonCopula(3.0), rc.GumbelCopula(2.5)], [rc.FrankCopula(4.0)]], structure="D"
    ),
    "C": VineCopula(
        [[rc.ClaytonCopula(3.0), rc.GumbelCopula(2.5)], [rc.FrankCopula(4.0)]], structure="C"
    ),
}


class TestTheGaussianIdentity:
    """A vine of Gaussian pair-copulas IS a Gaussian copula. Exactly."""

    @pytest.mark.parametrize("d", [3, 4, 5, 6, 7])
    @pytest.mark.parametrize("structure", ["C", "D"])
    def test_the_correlation_matrix_round_trips(self, d: int, structure: str) -> None:
        sigma = random_correlation(d, seed=d)
        recovered = gaussian_vine(sigma, structure).to_gaussian().sigma()
        assert np.allclose(recovered, sigma, atol=1e-12)

    @pytest.mark.parametrize("d", [3, 4, 5, 6, 7])
    @pytest.mark.parametrize("structure", ["C", "D"])
    def test_the_density_matches_the_gaussian_copula(self, d: int, structure: str) -> None:
        """The check that catches an error in the tree recursion at any depth.

        A partial-correlation bug that peeled the wrong element off the
        conditioning set was invisible at d = 3 -- where every conditioning set
        has at most one element -- and wrong by 0.11 in correlation at d = 4.
        """
        sigma = random_correlation(d, seed=d)
        vine = gaussian_vine(sigma, structure)
        points = np.random.default_rng(1).uniform(0.05, 0.95, size=(300, d))
        assert np.allclose(vine.logpdf(points), vine.to_gaussian().logpdf(points), atol=1e-9)

    def test_it_refuses_a_mixed_vine(self) -> None:
        with pytest.raises(ValueError, match="every pair-copula to be Gaussian"):
            MIXED["D"].to_gaussian()

    def test_the_implied_correlation_is_not_the_pair_parameter(self) -> None:
        """Tree 2 supplies a *partial* correlation, so 1-2 is implied, not given."""
        vine = VineCopula(
            [[rc.GaussianCopula(0.7), rc.GaussianCopula(0.4)], [rc.GaussianCopula(0.2)]],
            structure="C",
        )
        sigma = vine.to_gaussian().sigma()
        assert sigma[0, 1] == pytest.approx(0.7)
        assert sigma[0, 2] == pytest.approx(0.4)
        expected = 0.2 * np.sqrt((1 - 0.7**2) * (1 - 0.4**2)) + 0.7 * 0.4
        assert sigma[1, 2] == pytest.approx(expected)


class TestTheDensityIsADensity:
    @pytest.mark.parametrize("structure", ["C", "D"])
    def test_it_integrates_to_one(self, structure: str) -> None:
        vine = MIXED[structure]
        points = np.random.default_rng(0).uniform(size=(400_000, vine.dim))
        assert np.exp(vine.logpdf(points)).mean() == pytest.approx(1.0, abs=0.02)

    @pytest.mark.parametrize("structure", ["C", "D"])
    def test_it_is_finite_and_positive(self, structure: str) -> None:
        points = np.random.default_rng(0).uniform(0.001, 0.999, size=(5000, 3))
        density = MIXED[structure].pdf(points)
        assert np.all(np.isfinite(density))
        assert np.all(density > 0.0)

    def test_a_two_dimensional_vine_is_its_only_pair_copula(self) -> None:
        base = rc.ClaytonCopula(3.0)
        vine = VineCopula([[base]], structure="D")
        points = np.random.default_rng(0).uniform(0.02, 0.98, size=(200, 2))
        assert np.allclose(vine.logpdf(points), base.logpdf(points))

    def test_an_all_independence_vine_is_the_independence_copula(self) -> None:
        d = 4
        vine = VineCopula(
            [[rc.IndependenceCopula(2)] * (d - 1 - k) for k in range(d - 1)], structure="D"
        )
        points = np.random.default_rng(0).uniform(0.02, 0.98, size=(200, d))
        assert np.allclose(vine.logpdf(points), 0.0, atol=1e-12)

    def test_the_cdf_is_refused_with_a_useful_message(self) -> None:
        with pytest.raises(NotImplementedError, match=r"factorises the \*density\*"):
            MIXED["D"].cdf(np.full((1, 3), 0.5))


class TestTheSamplerAgreesWithTheDensity:
    @pytest.mark.parametrize("structure", ["C", "D"])
    def test_margins_are_uniform(self, structure: str) -> None:
        sample = MIXED[structure].rvs(60_000, random_state=0)
        for j in range(3):
            assert stats.kstest(sample[:, j], "uniform").pvalue > 1e-3

    @pytest.mark.parametrize("structure", ["C", "D"])
    def test_draws_stay_strictly_inside_the_cube(self, structure: str) -> None:
        sample = MIXED[structure].rvs(20_000, random_state=0)
        assert np.all((sample > 0.0) & (sample < 1.0))

    @pytest.mark.parametrize("structure", ["C", "D"])
    def test_tree_one_governs_the_pairs_it_joins(self, structure: str) -> None:
        """A D-vine's tree 1 pairs adjacent variables; a C-vine's pairs the root
        with each of the others."""
        vine = MIXED[structure]
        sample = vine.rvs(120_000, random_state=0)
        pairs = [(0, 1), (1, 2)] if structure == "D" else [(0, 1), (0, 2)]
        for (i, j), copula in zip(pairs, vine.pair_copulas[0], strict=True):
            observed = stats.kendalltau(sample[:, i], sample[:, j]).statistic
            assert observed == pytest.approx(copula.tau(), abs=0.01)

    @pytest.mark.parametrize("d", [3, 4, 5])
    def test_a_gaussian_vine_samples_the_right_correlation(self, d: int) -> None:
        """Sampler against `to_gaussian`, which the density already validates."""
        sigma = random_correlation(d, seed=d)
        sample = gaussian_vine(sigma, "D").rvs(200_000, random_state=0)
        scores = stats.norm.ppf(np.clip(sample, 1e-9, 1 - 1e-9))
        assert np.allclose(np.corrcoef(scores.T), sigma, atol=0.02)

    def test_sampling_is_reproducible(self) -> None:
        vine = MIXED["D"]
        assert np.array_equal(vine.rvs(50, random_state=7), vine.rvs(50, random_state=7))


class TestRosenblatt:
    """The forward direction of the sampler, and the tightest joint check."""

    def test_it_produces_independent_uniforms(self) -> None:
        vine = MIXED["D"]
        z = vine.rosenblatt(vine.rvs(20_000, random_state=0))
        for j in range(vine.dim):
            assert stats.kstest(z[:, j], "uniform").pvalue > 1e-3
        for i in range(vine.dim):
            for j in range(i + 1, vine.dim):
                assert abs(stats.kendalltau(z[:, i], z[:, j]).statistic) < 0.03

    def test_the_wrong_vine_leaves_visible_structure(self) -> None:
        """So the check above is not vacuous."""
        truth = MIXED["D"]
        wrong = VineCopula(
            [[rc.GumbelCopula(4.0), rc.ClaytonCopula(4.0)], [rc.FrankCopula(-6.0)]],
            structure="D",
        )
        z = wrong.rosenblatt(truth.rvs(20_000, random_state=0))
        worst = min(stats.kstest(z[:, j], "uniform").pvalue for j in range(3))
        assert worst < 1e-6

    def test_it_is_refused_for_a_c_vine(self) -> None:
        with pytest.raises(NotImplementedError, match="D-vines"):
            MIXED["C"].rosenblatt(np.full((1, 3), 0.5))


class TestStructure:
    def test_it_validates_the_tree_sizes(self) -> None:
        with pytest.raises(ValueError, match="needs 2 pair-copulas"):
            VineCopula([[rc.ClaytonCopula(2.0)], [rc.FrankCopula(2.0)]])

    def test_pair_copulas_must_be_bivariate(self) -> None:
        with pytest.raises(ValueError, match="must be bivariate"):
            VineCopula([[rc.ClaytonCopula(2.0, dim=3)]])

    def test_it_validates_the_structure_and_order(self) -> None:
        with pytest.raises(ValueError, match="structure must be"):
            VineCopula([[rc.ClaytonCopula(2.0)]], structure="R")
        with pytest.raises(ValueError, match="permutation"):
            VineCopula([[rc.ClaytonCopula(2.0)]], order=[0, 0])

    def test_the_order_permutes_the_variables(self) -> None:
        pairs = [[rc.ClaytonCopula(3.0), rc.GumbelCopula(2.5)], [rc.FrankCopula(4.0)]]
        plain = VineCopula(pairs, structure="D")
        permuted = VineCopula(pairs, structure="D", order=[2, 0, 1])
        sample = permuted.rvs(80_000, random_state=0)
        # Tree 1's first pair joins order[0] and order[1], i.e. variables 2 and 0.
        observed = stats.kendalltau(sample[:, 2], sample[:, 0]).statistic
        assert observed == pytest.approx(rc.ClaytonCopula(3.0).tau(), abs=0.01)
        assert plain.order != permuted.order

    def test_a_scalar_dependence_measure_is_refused(self) -> None:
        for method in (MIXED["D"].tau, MIXED["D"].rho, MIXED["D"].lambda_):
            with pytest.raises(NotImplementedError, match="pair"):
                method()

    def test_describe_names_every_edge_with_its_conditioning_set(self) -> None:
        text = MIXED["D"].describe()
        assert "0,1" in text and "1,2" in text and "0,2|1" in text
        assert "Clayton" in text and "Frank" in text

    def test_n_pairs(self) -> None:
        assert MIXED["D"].n_pairs == 3
        assert (
            VineCopula(
                [[rc.ClaytonCopula(2.0)] * 4]
                and [[rc.ClaytonCopula(2.0)] * (4 - k) for k in range(4)]
            ).n_pairs
            == 10
        )


class TestFitting:
    @pytest.mark.parametrize("structure", ["C", "D"])
    def test_it_recovers_the_families_of_tree_one(self, structure: str) -> None:
        truth = MIXED[structure]
        fitted = fit_vine(
            truth.rvs(4000, random_state=0),
            structure=structure,
            order=[0, 1, 2],
            families=["clayton", "gumbel", "frank", "gaussian"],
        )
        assert [c.name for c in fitted.pair_copulas[0]] == ["Clayton", "Gumbel"]

    @pytest.mark.parametrize("structure", ["C", "D"])
    def test_it_recovers_the_parameters(self, structure: str) -> None:
        truth = MIXED[structure]
        fitted = fit_vine(
            truth.rvs(4000, random_state=0),
            structure=structure,
            order=[0, 1, 2],
            families=["clayton", "gumbel", "frank"],
        )
        assert fitted.pair_copulas[0][0].params[0] == pytest.approx(3.0, rel=0.15)
        assert fitted.pair_copulas[0][1].params[0] == pytest.approx(2.5, rel=0.15)

    def test_the_fit_beats_a_misspecified_vine_on_likelihood(self) -> None:
        truth = MIXED["D"]
        data = truth.rvs(3000, random_state=0)
        fitted = fit_vine(data, structure="D", order=[0, 1, 2])
        wrong = VineCopula(
            [[rc.GumbelCopula(4.0), rc.ClaytonCopula(4.0)], [rc.FrankCopula(-6.0)]],
            structure="D",
        )
        assert fitted.loglik(data) > wrong.loglik(data)

    def test_it_fits_a_gaussian_vine_back_to_the_right_correlation(self) -> None:
        sigma = random_correlation(4, seed=11)
        data = gaussian_vine(sigma, "D").rvs(6000, random_state=0)
        fitted = fit_vine(data, structure="D", order=[0, 1, 2, 3], families=["gaussian"])
        assert np.allclose(fitted.to_gaussian().sigma(), sigma, atol=0.05)

    def test_truncation_sets_the_higher_trees_to_independence(self) -> None:
        """The standard way to stop a vine spending parameters on noise."""
        data = MIXED["D"].rvs(2000, random_state=0)
        fitted = fit_vine(data, structure="D", truncate=1)
        assert fitted.pair_copulas[1][0].name == "Independence"
        assert fitted.pair_copulas[0][0].name != "Independence"

    def test_the_default_order_puts_the_most_dependent_variable_first(self) -> None:
        rng = np.random.default_rng(0)
        hub = rng.uniform(size=3000)
        data = np.column_stack(
            [rng.uniform(size=3000), hub, np.clip(hub + rng.normal(0, 0.05, 3000), 0.001, 0.999)]
        )
        fitted = fit_vine(data, structure="C", families=["gaussian", "clayton"])
        assert fitted.order[0] in (1, 2)

    def test_it_needs_at_least_two_variables(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            fit_vine(np.random.default_rng(0).uniform(size=(50, 1)))


def _truncated_c_vine(d: int, rhos: np.ndarray, student: bool = False) -> VineCopula:
    pair = (
        (lambda r: rc.StudentCopula(float(r), df=5.0))
        if student
        else (lambda r: rc.GaussianCopula(float(r)))
    )
    trees = [[pair(r) for r in rhos]] + [
        [rc.IndependenceCopula(2)] * (d - 1 - k) for k in range(1, d - 1)
    ]
    return VineCopula(trees, structure="C")


class TestRosenblattOrder:
    """The transform comes back in the caller's column order, like rvs and logpdf."""

    def test_columns_are_in_original_variable_order(self) -> None:
        trees = [
            [rc.ClaytonCopula(2.0), rc.GumbelCopula(2.5), rc.FrankCopula(3.0)],
            [rc.FrankCopula(2.0), rc.GaussianCopula(0.3)],
            [rc.ClaytonCopula(0.5)],
        ]
        order = [2, 0, 3, 1]
        vine = VineCopula(trees, structure="D", order=order)
        u = vine.rvs(500, random_state=3)
        z = vine.rosenblatt(u)
        # The first variable on the path is passed through unchanged, in its own column.
        np.testing.assert_array_equal(z[:, order[0]], u[:, order[0]])
        # rvs maps uniform column i to variable order[i]; rosenblatt undoes exactly that.
        w = np.random.default_rng(3).uniform(size=(500, 4))
        np.testing.assert_allclose(z[:, order], w, atol=1e-7)

    def test_identity_order_is_unchanged(self) -> None:
        vine = MIXED["D"]
        u = vine.rvs(200, random_state=0)
        w = np.random.default_rng(0).uniform(size=(200, vine.dim))
        np.testing.assert_allclose(vine.rosenblatt(u), w, atol=1e-7)


class TestTruncation:
    """Independence trees past the truncation level are skipped, not walked."""

    def test_truncation_level(self) -> None:
        assert MIXED["D"].truncation_level == 2
        assert _truncated_c_vine(5, np.full(4, 0.5)).truncation_level == 1
        indep = VineCopula([[rc.IndependenceCopula(2)] * 2, [rc.IndependenceCopula(2)]])
        assert indep.truncation_level == 0
        u = indep.rvs(100, random_state=0)
        np.testing.assert_array_equal(u, np.random.default_rng(0).uniform(size=(100, 3)))
        np.testing.assert_array_equal(indep.logpdf(u), np.zeros(100))

    def test_a_large_truncated_vine_samples_fast(self) -> None:
        import time

        d = 121
        rhos = np.random.default_rng(0).uniform(0.3, 0.7, d - 1)
        vine = _truncated_c_vine(d, rhos, student=True)
        start = time.perf_counter()
        u = vine.rvs(4000, random_state=1)
        # Walking all 7,140 edges took minutes; one tree takes well under a second
        # on an idle machine.
        assert time.perf_counter() - start < 30.0
        assert u.shape == (4000, d)

    def test_truncated_c_vine_has_the_implied_one_factor_correlations(self) -> None:
        d = 6
        rhos = np.array([0.8, 0.6, 0.4, 0.7, 0.5])
        vine = _truncated_c_vine(d, rhos)
        z = stats.norm.ppf(vine.rvs(40_000, random_state=2))
        corr = np.corrcoef(z, rowvar=False)
        loadings = np.concatenate([[1.0], rhos])
        implied = np.outer(loadings, loadings)
        np.fill_diagonal(implied, 1.0)
        assert np.max(np.abs(corr - implied)) < 0.02

    @pytest.mark.parametrize("structure", ["C", "D"])
    def test_truncated_matches_explicit_zero_gaussians(self, structure: str) -> None:
        """Replace the independence trees by Gaussian(0) -- the same copula, but
        not recognised as truncated -- and every recursion must agree."""
        first = [rc.ClaytonCopula(2.0), rc.StudentCopula(0.5, df=4.0), rc.FrankCopula(3.0)]
        truncated = VineCopula(
            [first, [rc.IndependenceCopula(2)] * 2, [rc.IndependenceCopula(2)]],
            structure=structure,
        )
        explicit = VineCopula(
            [first, [rc.GaussianCopula(0.0)] * 2, [rc.GaussianCopula(0.0)]],
            structure=structure,
        )
        u = explicit.rvs(400, random_state=4)
        np.testing.assert_allclose(truncated.logpdf(u), explicit.logpdf(u), atol=1e-8)
        np.testing.assert_allclose(
            truncated.rvs(400, random_state=4), explicit.rvs(400, random_state=4), atol=1e-6
        )
        if structure == "D":
            np.testing.assert_allclose(truncated.rosenblatt(u), explicit.rosenblatt(u), atol=1e-8)

    @pytest.mark.parametrize(
        ("structure", "expected"),
        [
            (
                "C",
                [
                    [0.12857020276919962, 0.14579860771979763, 0.22594389876228577],
                    [0.028689008371944547, 0.019483803416669374, 0.16676686816953334],
                ],
            ),
            (
                "D",
                [
                    [0.12857020276919962, 0.14579860771979763, 0.17433057348688547],
                    [0.028689008371944547, 0.019483803416669374, 0.39407384294967784],
                ],
            ),
        ],
    )
    def test_untruncated_seeded_draws_are_unchanged(
        self, structure: str, expected: list[list[float]]
    ) -> None:
        """A full vine keeps its exact algorithm: same seed, same draws. Values
        pinned from rcopula 0.2.0 (bit-identical on the machine that pinned them;
        the tolerance only allows for another platform's libm)."""
        u = MIXED[structure].rvs(2, random_state=11)
        np.testing.assert_allclose(u, np.array(expected), rtol=1e-13, atol=0)

    @pytest.mark.parametrize("structure", ["C", "D"])
    def test_truncated_fit_has_the_right_shape(self, structure: str) -> None:
        u = MIXED["D"].rvs(400, random_state=0)
        u = np.column_stack([u, u[:, 0] * 0.5 + 0.25])
        fitted = fit_vine(u, structure=structure, truncate=1, families=["gaussian", "clayton"])
        assert [len(level) for level in fitted.pair_copulas] == [3, 2, 1]
        assert fitted.truncation_level <= 1


class TestDefaultOrder:
    def test_d_vine_default_order_is_the_tau_ranking(self) -> None:
        """The docstring promises the strength ordering for D-vines too."""
        from rcopula.vine import _default_order

        u = MIXED["C"].rvs(800, random_state=0)
        fitted = fit_vine(u, structure="D", families=["gaussian"])
        assert list(fitted.order) == _default_order(u, "D")
        tau = np.abs(rc.cor_kendall(u)).sum(axis=1)
        assert list(fitted.order) == [int(j) for j in np.argsort(-tau)]
