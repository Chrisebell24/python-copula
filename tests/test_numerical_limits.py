"""Regression tests for the numerical failures a parameter sweep exposed.

Every family here was correct over the parameter range the earlier tests
exercised and wrong outside it. The failures were not near-misses -- densities
came out negative, half a sample landed on the boundary of the unit square, a
CDF's margins were off by 0.475, Kendall's tau came back as 14.6 -- so this file
sweeps each family across its *whole* admissible range and checks the properties
that define a copula, rather than spot values.

The four axioms checked throughout:

* ``C(u, 1, ..., 1) = u`` in every coordinate;
* the density is finite and non-negative wherever it exists;
* draws lie strictly inside the unit cube, with uniform margins;
* the sample Kendall tau matches the population value.
"""

from __future__ import annotations

from itertools import pairwise
from typing import ClassVar

import numpy as np
import pytest
from scipy import stats

import rcopula as rc
from rcopula.core.measures import _tensor, rho_by_quadrature, tau_by_partials

GRID = np.random.default_rng(0).uniform(0.001, 0.999, size=(2000, 2))


def margin_error(cop: rc.Copula, coordinate: int = 0) -> float:
    """max |C(..., u_j, ...) - u| with every other coordinate at 1."""
    u = np.linspace(0.01, 0.99, 25)
    pts = np.ones((u.size, cop.dim))
    pts[:, coordinate] = u
    return float(np.max(np.abs(cop.cdf(pts) - u)))


class TestFrankAtLargeTheta:
    """``psi`` and ``ipsi`` both saturated, in opposite directions.

    ``psi`` formed ``1 - h e^{-t}`` by subtraction; past ``theta = 37`` both
    factors round to 1 and the difference is exactly zero, so the generator
    returned ``inf`` and stopped inverting itself. ``ipsi`` formed a ratio that
    saturates to exactly ``-1`` for *every* negative theta below about -40, with
    ``log1p(-1) = -inf``. Between them the CDF's margins were wrong by 0.22 at
    ``theta = 50`` -- which is only ``tau = 0.92``, well inside the range people
    fit.
    """

    THETAS: ClassVar[list[float]] = [-700.0, -300.0, -100.0, -50.0, -20.0, 20.0, 50.0, 100.0, 700.0]

    @pytest.mark.parametrize("theta", THETAS)
    def test_the_generator_inverts_itself(self, theta: float) -> None:
        gen = rc.FrankCopula(theta).generator
        u = np.array([1e-9, 1e-6, 0.02, 0.3, 0.5, 0.9, 0.98, 1 - 1e-9])
        assert np.allclose(gen.psi(gen.ipsi(u, theta), theta), u, atol=1e-14)

    @pytest.mark.parametrize("theta", THETAS)
    def test_margins_are_uniform(self, theta: float) -> None:
        assert margin_error(rc.FrankCopula(theta)) < 1e-12

    @pytest.mark.parametrize("theta", THETAS)
    def test_the_density_is_finite_and_non_negative(self, theta: float) -> None:
        """Beyond |theta| ~ 700 the generator's argument itself underflows.

        That is the double-precision floor rather than a formula problem, and it
        is far past any parameter worth fitting -- theta = 700 is already
        tau = 0.994. Inside the range, the density must be clean.
        """
        pdf = rc.FrankCopula(theta).pdf(GRID)
        assert np.all(np.isfinite(pdf))
        assert np.all(pdf >= 0.0)

    @pytest.mark.parametrize("theta", [-100.0, -50.0, 50.0, 100.0])
    def test_the_density_is_the_derivative_of_the_cdf(self, theta: float) -> None:
        cop = rc.FrankCopula(theta)
        h = 1e-4
        for x, y in [(0.3, 0.4), (0.5, 0.5), (0.2, 0.25), (0.45, 0.5)]:
            analytic = cop.pdf([[x, y]])[0]
            if analytic < 1e-6:
                continue  # differencing the CDF is pure rounding noise there
            corners = np.array([[x + h, y + h], [x + h, y - h], [x - h, y + h], [x - h, y - h]])
            c = cop.cdf(corners)
            numeric = (c[0] - c[1] - c[2] + c[3]) / (4 * h * h)
            assert analytic == pytest.approx(numeric, rel=1e-4)


class TestFrankSampling:
    """A quarter of the draws landed on the boundary, and the rest were wrong.

    Kemp's inversion divides by ``log(q)`` with ``q = 1 - e^{v log(1-p)}``. Once
    the exponent drops below about -37 that rounds to exactly 1, ``log(q)`` is
    exactly 0, and the variate is ``inf`` -- so ``u = psi(E/inf) = 1``. At
    ``theta = 50`` it hit 25% of draws; at ``theta = 100``, 62%. The KS test
    against uniform returned p = 0.
    """

    @pytest.mark.parametrize("theta", [20.0, 50.0, 100.0, 300.0])
    def test_no_draw_lands_on_the_boundary(self, theta: float) -> None:
        sample = rc.FrankCopula(theta).rvs(60_000, random_state=0)
        assert np.all((sample > 0.0) & (sample < 1.0))

    @pytest.mark.parametrize("theta", [20.0, 50.0, 100.0, 300.0, -50.0, -100.0])
    def test_margins_are_uniform(self, theta: float) -> None:
        sample = rc.FrankCopula(theta).rvs(60_000, random_state=0)
        for j in range(2):
            assert stats.kstest(sample[:, j], "uniform").pvalue > 1e-4

    @pytest.mark.parametrize("theta", [20.0, 50.0, 100.0, 300.0, -50.0, -100.0])
    def test_dependence_matches_theory(self, theta: float) -> None:
        cop = rc.FrankCopula(theta)
        sample = cop.rvs(60_000, random_state=0)
        assert stats.kendalltau(sample[:, 0], sample[:, 1]).statistic == pytest.approx(
            cop.tau(), abs=0.005
        )

    def test_the_log_series_sampler_survives_a_saturated_p(self) -> None:
        """``p = 1 - e^{-theta}`` is exactly 1 in float64 past theta ~ 37."""
        from rcopula.special.stable import rlog_series

        rng = np.random.default_rng(0)
        theta = 60.0
        p = -np.expm1(-theta)
        assert p == 1.0
        draws = rlog_series(10_000, p, rng, log1mp=-theta)
        assert np.all(np.isfinite(draws))
        assert np.all(draws >= 1.0)

    def test_it_refuses_a_saturated_p_without_the_exact_log(self) -> None:
        from rcopula.special.stable import rlog_series

        with pytest.raises(ValueError, match="pass log1mp explicitly"):
            rlog_series(10, 1.0, np.random.default_rng(0))


class TestClaytonAtLargeTheta:
    """``psi^{-1}(u) = u^{-theta} - 1`` overflows, and the frailty underflows.

    At ``theta = 1000`` a coordinate of 0.02 gives ``inf`` from the inverse
    generator, so the CDF's margins were wrong by 0.475 and the density was
    ``nan``. Independently, the frailty ``Gamma(1/theta, 1)`` returns exactly
    zero for about half the draws at that shape, sending ``t = E/V`` to infinity
    and half the sample to ``u = 0``.
    """

    THETAS: ClassVar[list[float]] = [1.0, 10.0, 100.0, 1000.0, 10_000.0]

    @pytest.mark.parametrize("theta", THETAS)
    @pytest.mark.parametrize("dim", [2, 3])
    def test_margins_are_uniform(self, theta: float, dim: int) -> None:
        assert margin_error(rc.ClaytonCopula(theta, dim=dim)) < 1e-12

    @pytest.mark.parametrize("theta", THETAS)
    @pytest.mark.parametrize("dim", [2, 3])
    def test_draws_stay_inside_the_cube(self, theta: float, dim: int) -> None:
        sample = rc.ClaytonCopula(theta, dim=dim).rvs(40_000, random_state=0)
        assert np.all((sample > 0.0) & (sample < 1.0))

    @pytest.mark.parametrize("theta", THETAS)
    def test_sampling_reproduces_tau(self, theta: float) -> None:
        cop = rc.ClaytonCopula(theta)
        sample = cop.rvs(40_000, random_state=0)
        assert stats.kendalltau(sample[:, 0], sample[:, 1]).statistic == pytest.approx(
            cop.tau(), abs=0.005
        )

    @pytest.mark.parametrize("theta", [0.5, 2.0, 8.0])
    def test_the_log_space_cdf_agrees_with_the_direct_formula(self, theta: float) -> None:
        """The stable route must not shift the answer where the direct one worked."""
        direct = (GRID[:, 0] ** -theta + GRID[:, 1] ** -theta - 1.0) ** (-1.0 / theta)
        assert np.allclose(rc.ClaytonCopula(theta).cdf(GRID), direct, rtol=1e-13)

    def test_the_negative_branch_still_uses_the_direct_form(self) -> None:
        """It must, because that is where C is exactly zero on a whole region."""
        for theta in (-0.5, -0.9):
            direct = np.maximum(GRID[:, 0] ** -theta + GRID[:, 1] ** -theta - 1.0, 0.0) ** (
                -1.0 / theta
            )
            assert np.allclose(rc.ClaytonCopula(theta).cdf(GRID), direct, atol=1e-14)

    def test_the_frailty_boost_avoids_underflow(self) -> None:
        """log V must stay finite where V itself is exactly zero."""
        gen = rc.ClaytonCopula(1000.0).generator
        rng = np.random.default_rng(0)
        assert np.mean(gen.rvs_frailty(10_000, 1000.0, rng) == 0.0) > 0.1
        assert np.all(np.isfinite(gen.rvs_log_frailty(10_000, 1000.0, rng)))


class TestMarshallOlkinDensity:
    """The two branches of the density were swapped.

    ``C = min(u^{1-a1} v, u v^{1-a2})``, and on ``u^{a1} > v^{a2}`` the active
    branch is ``u^{1-a1} v``, whose mixed derivative is ``(1-a1) u^{-a1}`` -- a
    function of *u*. The code returned the *v* expression there. With
    ``a1 = a2`` the two agree on the diagonal, which is why spot checks missed
    it; everywhere else the density was simply the wrong function.
    """

    ALPHAS: ClassVar[list[tuple[float, float]]] = [
        (0.05, 0.05),
        (0.3, 0.3),
        (0.2, 0.8),
        (0.8, 0.2),
        (0.7, 0.4),
        (0.95, 0.95),
    ]

    @pytest.mark.parametrize(("a1", "a2"), ALPHAS)
    def test_the_density_is_the_derivative_of_the_cdf(self, a1: float, a2: float) -> None:
        cop = rc.MarshallOlkinCopula(a1, a2)
        h = 1e-5
        for x, y in [(0.85, 0.462), (0.3, 0.8), (0.6, 0.2), (0.25, 0.35), (0.9, 0.55)]:
            if abs(x**a1 - y**a2) < 1e-2:
                continue  # the singular curve, where no density exists
            corners = np.array([[x + h, y + h], [x + h, y - h], [x - h, y + h], [x - h, y - h]])
            c = cop.cdf(corners)
            numeric = (c[0] - c[1] - c[2] + c[3]) / (4 * h * h)
            assert cop.pdf([[x, y]])[0] == pytest.approx(numeric, abs=1e-5)

    @pytest.mark.parametrize(("a1", "a2"), ALPHAS)
    def test_tau_and_rho_match_quadrature_of_the_cdf(self, a1: float, a2: float) -> None:
        """Both routes use only the CDF, so they see the singular mass too."""
        cop = rc.MarshallOlkinCopula(a1, a2)
        assert tau_by_partials(cop) == pytest.approx(cop.tau(), abs=3e-3)
        assert rho_by_quadrature(cop) == pytest.approx(cop.rho(), abs=3e-3)

    def test_the_density_is_asymmetric_when_the_parameters_are(self) -> None:
        """The swap was invisible whenever a1 == a2; this is the case that caught it."""
        cop = rc.MarshallOlkinCopula(0.2, 0.8)
        assert cop.pdf([[0.4, 0.6]])[0] != pytest.approx(cop.pdf([[0.6, 0.4]])[0], rel=0.01)


class TestExtremeValueCurvature:
    """Galambos' ``A''`` came out negative, which is impossible.

    ``A`` is convex by definition. The direct second derivative is a difference
    of two enormous terms and at ``theta = 30`` it returned ``-9.3e-12``, which
    drove the copula density negative and then ``nan`` for half the unit square.
    Expanding the difference collapses it to a single positive term.
    """

    THETAS: ClassVar[list[float]] = [0.1, 0.5, 2.0, 10.0, 30.0, 50.0, 100.0, 500.0]

    @pytest.mark.parametrize("theta", THETAS)
    def test_the_pickands_function_satisfies_its_defining_bounds(self, theta: float) -> None:
        cop = rc.GalambosCopula(theta)
        t = np.linspace(1e-6, 1 - 1e-6, 2001)
        a = cop.A(t)
        assert np.all(a >= np.maximum(t, 1 - t) - 1e-12)
        assert np.all(a <= 1.0 + 1e-12)

    @pytest.mark.parametrize("theta", THETAS)
    def test_the_pickands_function_is_convex(self, theta: float) -> None:
        t = np.linspace(1e-6, 1 - 1e-6, 2001)
        assert np.all(rc.GalambosCopula(theta).d2A(t) >= 0.0)

    @pytest.mark.parametrize("theta", THETAS)
    def test_the_derivatives_are_consistent_with_each_other(self, theta: float) -> None:
        """Chained by integration rather than by finite differences.

        ``int_a^b A' = A(b) - A(a)`` ties A' to A, and
        ``int_a^b A'' = A'(b) - A'(a)`` ties A'' to A'. Differencing would test
        the yardstick instead: A develops a kink at ``t = 1/2`` whose width
        shrinks like ``1/theta``, so a step small enough to resolve it at
        ``theta = 500`` is swamped by rounding, and one large enough to avoid
        rounding straddles the kink -- at ``theta = 10`` the difference misses
        A' by 4e-5 and A'' by 13%. Both identities hold to 1e-12 at every theta.
        """
        cop = rc.GalambosCopula(theta)
        a, b = 0.05, 0.95
        nodes, weights = np.polynomial.legendre.leggauss(6000)
        grid = 0.5 * (b - a) * nodes + 0.5 * (a + b)
        scaled = 0.5 * (b - a) * weights

        first = float(np.sum(scaled * cop.dA(grid)))
        assert first == pytest.approx(
            float(cop.A(np.array([b]))[0] - cop.A(np.array([a]))[0]), abs=1e-10
        )

        second = float(np.sum(scaled * cop.d2A(grid)))
        assert second == pytest.approx(
            float(cop.dA(np.array([b]))[0] - cop.dA(np.array([a]))[0]), abs=1e-10
        )

    @pytest.mark.parametrize(
        "cop",
        [
            *(rc.GalambosCopula(t) for t in [0.1, 2.0, 10.0, 50.0, 500.0]),
            *(rc.HuslerReissCopula(t) for t in [0.1, 2.0, 10.0, 50.0, 500.0]),
            *(rc.TawnCopula(t) for t in [0.0, 0.5, 1.0]),
            *(rc.TEVCopula(r, df=4.0) for r in [-0.9, 0.0, 0.95]),
        ],
    )
    def test_the_density_is_finite_and_non_negative(self, cop: rc.Copula) -> None:
        pdf = cop.pdf(GRID)
        assert np.all(np.isfinite(pdf))
        assert np.all(pdf >= 0.0)

    @pytest.mark.parametrize("theta", [0.5, 2.0, 10.0, 50.0])
    def test_the_density_is_the_derivative_of_the_cdf(self, theta: float) -> None:
        cop = rc.GalambosCopula(theta)
        h = 1e-6
        for x in (0.2, 0.4, 0.6, 0.8):
            corners = np.array([[x + h, x + h], [x + h, x - h], [x - h, x + h], [x - h, x - h]])
            c = cop.cdf(corners)
            numeric = (c[0] - c[1] - c[2] + c[3]) / (4 * h * h)
            assert cop.pdf([[x, x]])[0] == pytest.approx(numeric, rel=1e-3)


class TestPlackettTau:
    """Kendall's tau came back as 14.6 -- outside the range it can occupy.

    ``4 int int C c - 1`` needs the density, which under strong dependence
    concentrates onto the diagonal faster than any quadrature can follow. The
    equivalent ``1 - 4 int int dC/du dC/dv`` has an integrand bounded in
    ``[0, 1]`` and holds up across nine orders of magnitude in theta.
    """

    @pytest.mark.parametrize("theta", [1e-3, 1e-2, 0.5, 1.0, 2.0, 10.0, 100.0, 1e4, 1e6, 1e8])
    def test_tau_stays_in_range_and_matches_the_sample(self, theta: float) -> None:
        cop = rc.PlackettCopula(theta)
        assert -1.0 <= cop.tau() <= 1.0
        sample = cop.rvs(200_000, random_state=0)
        assert cop.tau() == pytest.approx(
            stats.kendalltau(sample[:, 0], sample[:, 1]).statistic, abs=0.005
        )

    def test_tau_is_monotone_in_theta(self) -> None:
        thetas = [1e-3, 1e-2, 0.5, 1.0, 2.0, 10.0, 100.0, 1e4, 1e6]
        taus = [rc.PlackettCopula(t).tau() for t in thetas]
        assert all(a < b for a, b in pairwise(taus))

    def test_independence_gives_exactly_zero(self) -> None:
        assert rc.PlackettCopula(1.0).tau() == pytest.approx(0.0, abs=1e-9)


class TestStudentSmallDegreesOfFreedom:
    """The radial density's ``s^{df-1}`` factor is non-smooth below df = 2.

    Below 1 it diverges at the origin; between 1 and 2 its derivative does.
    Either way Gauss-Legendre loses mass -- at ``df = 0.5`` it lost 0.44% of it,
    so every CDF value came out short by that constant and ``C(u, 1) = u``
    failed by 4e-3.
    """

    @pytest.mark.parametrize("df", [0.1, 0.25, 0.5, 0.99, 1.0, 1.5, 1.99, 2.0, 4.0, 30.0])
    @pytest.mark.parametrize("rho", [-0.9, 0.0, 0.5])
    def test_margins_are_uniform(self, df: float, rho: float) -> None:
        assert margin_error(rc.StudentCopula(rho, df=df)) < 1e-6

    @pytest.mark.parametrize("df", [0.5, 4.0])
    def test_margins_are_uniform_in_three_dimensions(self, df: float) -> None:
        cop = rc.StudentCopula(0.4, dim=3, df=df)
        assert all(margin_error(cop, j) < 1e-6 for j in range(3))

    @pytest.mark.parametrize("df", [0.25, 0.5, 1.0, 4.0])
    def test_the_cdf_matches_simulation(self, df: float) -> None:
        cop = rc.StudentCopula(0.5, df=df)
        sample = cop.rvs(400_000, random_state=0)
        pts = np.array([[0.2, 0.3], [0.5, 0.5], [0.8, 0.6], [0.1, 0.9]])
        empirical = np.array([np.mean((sample[:, 0] <= a) & (sample[:, 1] <= b)) for a, b in pts])
        assert np.allclose(cop.cdf(pts), empirical, atol=3e-3)

    @pytest.mark.parametrize("df", [0.5, 2.0, 4.0])
    def test_the_cdf_is_monotone(self, df: float) -> None:
        cop = rc.StudentCopula(0.5, df=df)
        u = np.linspace(0.02, 0.98, 40)
        values = cop.cdf(np.column_stack([u, np.full_like(u, 0.6)]))
        assert np.all(np.diff(values) > 0)


class TestSingularComponents:
    """Which copulas carry mass no density represents -- and how much.

    Worth pinning, because a density that integrates to less than one looks like
    a bug and is sometimes the correct answer. The two cases here are opposite:
    Marshall-Olkin genuinely puts mass on a curve, while Clayton below zero does
    not and merely has a density that diverges near the edge of its support.
    """

    @pytest.mark.parametrize(("a1", "a2"), [(0.3, 0.3), (0.7, 0.7), (0.2, 0.8), (0.9, 0.4)])
    def test_marshall_olkin_singular_mass_equals_tau(self, a1: float, a2: float) -> None:
        """A closed-form identity, and a sharp check on the density.

        The mass Marshall-Olkin places on ``u^{a1} = v^{a2}`` is exactly
        ``a1 a2 / (a1 + a2 - a1 a2)``, which is also its Kendall's tau. So the
        density -- the absolutely continuous part alone -- must integrate to
        ``1 - tau``. Getting the density's *branches* wrong changes that number,
        which is how the swapped branches would have been caught.
        """
        cop = rc.MarshallOlkinCopula(a1, a2)
        points, weights = _tensor(160)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            continuous = float(np.sum(weights * np.nan_to_num(cop.pdf(points))))
        assert continuous == pytest.approx(1.0 - cop.tau(), abs=5e-3)

    @pytest.mark.parametrize("theta", [-0.05, -0.2, -0.5])
    def test_clayton_below_zero_has_no_singular_component(self, theta: float) -> None:
        """Its density integrates to one; the support boundary is not an atom.

        Below the curve ``u^-theta + v^-theta = 1`` the copula is identically
        zero, which looks like the setup for singular mass. There is none: the
        conditional distribution approaches zero *continuously* at the boundary,
        because the exponent ``-1/theta - 1`` is positive throughout
        ``(-1, 0)``. What does happen is that the density diverges there like
        ``s^{-1/theta - 2}``, so as theta approaches -1 no quadrature resolves
        it -- which is a statement about integration, not about the copula.
        """
        points, weights = _tensor(160)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            mass = float(np.sum(weights * np.nan_to_num(rc.ClaytonCopula(theta).pdf(points))))
        assert mass == pytest.approx(1.0, abs=5e-3)

    @pytest.mark.parametrize("cop", [rc.FrechetUpperCopula(2), rc.FrechetLowerCopula(2)])
    def test_the_frechet_bounds_refuse_a_density(self, cop: rc.Copula) -> None:
        """Purely singular, so raising is the honest answer."""
        with pytest.raises(NotImplementedError, match="singular"):
            cop.pdf([[0.3, 0.4]])
