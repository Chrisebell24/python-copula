r"""Validation against the literature, with no reference to R.

Everything in ``tests/test_golden_*.py`` asks the same question: *does this agree
with R?* That is a useful question and an insufficient one -- agreeing with R
would also mean faithfully reproducing an R bug. This module asks a different
one: **does this agree with the definition?**

Nothing here consults a golden fixture. Each check evaluates a quantity from its
published definition -- a double integral, a limit, a frailty construction --
and compares it against the closed form the package returns. The two are
computed by completely different routes, so agreeing to eleven digits is
evidence about both.

The definitions used:

===================================  =========================================
:math:`\rho_S = 12\iint C - 3`       Nelsen (2006), Theorem 5.1.6
:math:`\tau = 1 - 4\iint C_u C_v`    Nelsen (2006), Corollary 5.1.2
:math:`\beta = 4C(1/2,1/2) - 1`      Blomqvist (1950)
:math:`\lambda_L = \lim_{u\to0^+} C(u,u)/u`   Joe (1997), Section 2.1.10
:math:`\lambda_U = \lim_{u\to1^-} \frac{1-2u+C(u,u)}{1-u}`  same
:math:`W \le C \le M`                Frechet (1951), Hoeffding (1940)
===================================  =========================================

plus the structural identities that connect different parts of the theory:
gamma frailty gives Clayton (Clayton 1978; Oakes 1989), positive-stable frailty
gives Gumbel (Hougaard 1986), :math:`\tau = \frac{2}{\pi}\arcsin\rho` for
*every* elliptical copula (Lindskog, McNeil and Schmock 2003), the Nataf
transform is a Gaussian copula (Lebrun and Dutfoy 2009), and mutual information
is copula entropy (Ma and Sun 2011).

Two constants are pinned against ``mpmath`` at 25 digits rather than against any
implementation in this package.

References
----------
Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed. Springer.
Joe, H. (1997). *Multivariate Models and Dependence Concepts*. Chapman & Hall.
Blomqvist, N. (1950). On a measure of dependence between two random variables.
    *Annals of Mathematical Statistics* 21(4), 593-600.
Lindskog, F., McNeil, A. and Schmock, U. (2003). Kendall's tau for elliptical
    distributions. In *Credit Risk*, 149-156. Physica-Verlag.
Clayton, D. G. (1978). A model for association in bivariate life tables.
    *Biometrika* 65(1), 141-151.
Hougaard, P. (1986). A class of multivariate failure time distributions.
    *Biometrika* 73(3), 671-678.
Lebrun, R. and Dutfoy, A. (2009). An innovating analysis of the Nataf
    transformation from the copula viewpoint.
    *Probabilistic Engineering Mechanics* 24(3), 312-320.
Ma, J. and Sun, Z. (2011). Mutual information is copula entropy.
    *Tsinghua Science and Technology* 16(1), 51-54.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy import stats

import rcopula as rc
from rcopula import conditional_cdf

#: Every bivariate family, at a parameter well inside its range. The point of
#: the list is coverage: a definition that holds for six families and fails for
#: the seventh is exactly the case a hand-picked example would miss.
BIVARIATE = [
    rc.IndependenceCopula(2),
    rc.GaussianCopula(0.5),
    rc.GaussianCopula(-0.7),
    rc.StudentCopula(0.5, df=4.0),
    rc.StudentCopula(-0.3, df=8.0),
    rc.ClaytonCopula(2.0),
    rc.ClaytonCopula(-0.5),
    rc.GumbelCopula(2.0),
    rc.FrankCopula(5.0),
    rc.FrankCopula(-3.0),
    rc.JoeCopula(2.0),
    rc.AMHCopula(0.7),
    rc.AMHCopula(-0.6),
    rc.PlackettCopula(4.0),
    rc.FGMCopula(0.5),
    rc.GalambosCopula(1.0),
    rc.HuslerReissCopula(1.5),
    rc.TawnCopula(0.6),
    rc.TEVCopula(0.5, df=4.0),
    rc.MarshallOlkinCopula([0.4, 0.7]),
    rc.RotatedCopula(rc.ClaytonCopula(2.0), 90),
    rc.KhoudrajiCopula(rc.IndependenceCopula(2), rc.GumbelCopula(4.0), shapes=(0.4, 0.95)),
    rc.MixtureCopula([rc.ClaytonCopula(3.0), rc.GumbelCopula(2.0)], weights=[0.4, 0.6]),
]

#: Families whose CDF has a kink or a singular component, where tensor
#: Gauss-Legendre cannot reach machine precision. They are still checked, just
#: at the accuracy the geometry allows.
NON_SMOOTH = {"MarshallOlkinCopula"}


def _identifier(copula: rc.Copula) -> str:
    params = "-".join(f"{p:g}" for p in np.atleast_1d(copula.params) if np.isfinite(p))
    return f"{type(copula).__name__}{params}"


def _tolerance(copula: rc.Copula, smooth: float, rough: float) -> float:
    return rough if type(copula).__name__ in NON_SMOOTH else smooth


def _grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights mapped onto (0, 1)."""
    nodes, weights = leggauss(n)
    return 0.5 * (nodes + 1.0), 0.5 * weights


class TestSpearmanFromItsDefinition:
    r""":math:`\rho_S = 12 \iint_{[0,1]^2} C(u,v)\,du\,dv - 3`.

    The copula's *distribution function* is bounded and smooth on the interior,
    so this integral is far better conditioned than the density version and
    tensor quadrature reaches eleven digits.
    """

    @staticmethod
    def integrate(copula: rc.Copula, n: int = 240) -> float:
        u, w = _grid(n)
        first, second = np.meshgrid(u, u, indexing="ij")
        values = np.asarray(copula.cdf(np.column_stack([first.ravel(), second.ravel()]))).reshape(
            n, n
        )
        return 12.0 * float(w @ values @ w) - 3.0

    @pytest.mark.parametrize("copula", BIVARIATE, ids=_identifier)
    def test_matches_the_closed_form(self, copula: rc.Copula) -> None:
        assert self.integrate(copula) == pytest.approx(
            copula.rho(), abs=_tolerance(copula, 1e-9, 1e-5)
        )

    def test_the_bounds_attain_plus_and_minus_one(self) -> None:
        # Both bounds have a kink along a diagonal. Polynomial quadrature
        # converges at O(n^-2) against a kink instead of exponentially, so 240
        # nodes buy four digits here where they buy eleven for a smooth family.
        # The exact values are trivial -- int int min(u,v) = 1/3 -- so this is a
        # statement about the quadrature, kept because the bounds belong in the
        # same table as everything else.
        assert self.integrate(rc.FrechetUpperCopula(2)) == pytest.approx(1.0, abs=1e-4)
        assert self.integrate(rc.FrechetLowerCopula(2)) == pytest.approx(-1.0, abs=1e-4)
        assert rc.FrechetUpperCopula(2).rho() == pytest.approx(1.0, abs=1e-12)
        assert rc.FrechetLowerCopula(2).rho() == pytest.approx(-1.0, abs=1e-12)

    def test_gaussian_matches_its_arcsine_formula(self) -> None:
        # rho_S = (6/pi) arcsin(rho/2), which is a closed form the integral
        # knows nothing about.
        for rho in (-0.9, -0.4, 0.0, 0.25, 0.8):
            expected = 6.0 / np.pi * np.arcsin(rho / 2.0)
            assert self.integrate(rc.GaussianCopula(rho)) == pytest.approx(expected, abs=1e-9)

    def test_fgm_is_theta_over_three(self) -> None:
        for theta in (-1.0, -0.3, 0.4, 1.0):
            assert self.integrate(rc.FGMCopula(theta)) == pytest.approx(theta / 3.0, abs=1e-12)


class TestKendallFromItsDefinition:
    r""":math:`\tau = 1 - 4 \iint \partial_u C \cdot \partial_v C \,du\,dv`.

    Nelsen's Corollary 5.1.2. It uses only the two h-functions, so it is
    independent of however the package chooses to compute tau -- which for
    several families is a completely different integral.
    """

    @staticmethod
    def integrate(copula: rc.Copula, n: int = 300) -> float:
        u, w = _grid(n)
        first, second = np.meshgrid(u, u, indexing="ij")
        points = np.column_stack([first.ravel(), second.ravel()])
        du = np.asarray(conditional_cdf(copula, points, 0)).reshape(n, n)
        dv = np.asarray(conditional_cdf(copula, points, 1)).reshape(n, n)
        return 1.0 - 4.0 * float(w @ (du * dv) @ w)

    @pytest.mark.parametrize("copula", BIVARIATE, ids=_identifier)
    def test_matches_the_closed_form(self, copula: rc.Copula) -> None:
        # Elliptical h-functions go through a numerical route, hence the looser
        # smooth tolerance here than for Spearman.
        assert self.integrate(copula) == pytest.approx(
            copula.tau(), abs=_tolerance(copula, 1e-6, 1e-4)
        )

    def test_gumbel_inverts_to_one_over_one_minus_tau(self) -> None:
        # The identity every worked example in the literature opens with.
        for tau in (0.25, 0.5, 0.75, 0.05, 0.9):
            copula = rc.GumbelCopula.from_tau(tau)
            assert copula.params[0] == pytest.approx(1.0 / (1.0 - tau), rel=1e-12)
            assert self.integrate(copula) == pytest.approx(tau, abs=1e-6)

    def test_clayton_is_theta_over_theta_plus_two(self) -> None:
        for theta in (-0.5, 0.5, 2.0, 8.0):
            # Strong dependence piles the h-functions up against the diagonal,
            # so the quadrature error grows with theta -- 3e-8 at theta = 8.
            assert self.integrate(rc.ClaytonCopula(theta)) == pytest.approx(
                theta / (theta + 2.0), abs=1e-6
            )

    def test_fgm_is_two_theta_over_nine(self) -> None:
        for theta in (-1.0, 0.4, 1.0):
            assert self.integrate(rc.FGMCopula(theta)) == pytest.approx(
                2.0 * theta / 9.0, abs=1e-10
            )


class TestEllipticalArcsineLaw:
    r""":math:`\tau = \frac{2}{\pi}\arcsin\rho` for **every** elliptical copula.

    Lindskog, McNeil and Schmock: the result depends only on radial symmetry, so
    the degrees of freedom cancel entirely. Spearman's rho does *not* have that
    property, which is the asymmetry worth pinning down -- the two measures are
    often described as interchangeable and here they demonstrably are not.
    """

    @pytest.mark.parametrize("rho", [-0.95, -0.5, -0.1, 0.0, 0.3, 0.7, 0.99])
    def test_kendall_does_not_depend_on_the_degrees_of_freedom(self, rho: float) -> None:
        expected = 2.0 / np.pi * np.arcsin(rho)
        assert rc.GaussianCopula(rho).tau() == pytest.approx(expected, abs=1e-12)
        for df in (1.0, 2.5, 4.0, 30.0, 200.0):
            assert rc.StudentCopula(rho, df=df).tau() == pytest.approx(expected, abs=1e-12)

    def test_spearman_does_depend_on_it(self) -> None:
        gaussian = rc.GaussianCopula(0.5).rho()
        heavy = rc.StudentCopula(0.5, df=3.0).rho()
        assert gaussian != pytest.approx(heavy, abs=1e-4)
        # The t copula's Spearman rho sits below the Gaussian one at the same
        # linear correlation: its extra mass in the corners is *joint*, which
        # concordance in the middle does not see.
        assert heavy < gaussian

    def test_the_t_copula_approaches_the_gaussian_one(self) -> None:
        for df in (50.0, 200.0, 1000.0):
            difference = abs(rc.StudentCopula(0.6, df=df).rho() - rc.GaussianCopula(0.6).rho())
            assert difference < 60.0 / df


class TestBlomqvistFromItsDefinition:
    r""":math:`\beta = 4\,C(1/2, 1/2) - 1`. No integral, no limit -- one CDF
    evaluation, so any disagreement is unambiguous."""

    @pytest.mark.parametrize("copula", BIVARIATE, ids=_identifier)
    def test_matches_the_closed_form(self, copula: rc.Copula) -> None:
        centre = float(np.asarray(copula.cdf([[0.5, 0.5]]))[0])
        assert copula.beta() == pytest.approx(4.0 * centre - 1.0, abs=1e-10)

    def test_gaussian_is_the_orthant_probability(self) -> None:
        # beta = (2/pi) arcsin(rho): the same arcsine law, at the median.
        for rho in (-0.8, -0.2, 0.4, 0.9):
            assert rc.GaussianCopula(rho).beta() == pytest.approx(
                2.0 / np.pi * np.arcsin(rho), abs=1e-9
            )

    def test_the_bounds_attain_plus_and_minus_one(self) -> None:
        assert rc.FrechetUpperCopula(2).beta() == pytest.approx(1.0, abs=1e-12)
        assert rc.FrechetLowerCopula(2).beta() == pytest.approx(-1.0, abs=1e-12)


class TestTailDependenceFromItsDefinition:
    r""":math:`\lambda_L = \lim_{u\to0^+} C(u,u)/u` and
    :math:`\lambda_U = \lim_{u\to1^-} (1-2u+C(u,u))/(1-u)`.

    Evaluated along a sequence approaching the corner. A limit cannot be
    asserted at a point, so the test is that the sequence *converges to* the
    closed form, which is a stronger statement than agreeing at one radius.
    """

    @staticmethod
    def lower_sequence(copula: rc.Copula, radii: np.ndarray) -> np.ndarray:
        points = np.column_stack([radii, radii])
        return np.asarray(copula.cdf(points)) / radii

    @staticmethod
    def upper_sequence(copula: rc.Copula, radii: np.ndarray) -> np.ndarray:
        u = 1.0 - radii
        points = np.column_stack([u, u])
        return (1.0 - 2.0 * u + np.asarray(copula.cdf(points))) / radii

    RADII: ClassVar[np.ndarray] = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6])

    @pytest.mark.parametrize("copula", BIVARIATE, ids=_identifier)
    def test_lower_tail_converges_to_the_closed_form(self, copula: rc.Copula) -> None:
        target = copula.lambda_().lower
        sequence = self.lower_sequence(copula, self.RADII)
        assert abs(sequence[-1] - target) < 0.02
        # And it is genuinely converging, not merely close at one radius.
        assert abs(sequence[-1] - target) <= abs(sequence[0] - target) + 1e-9

    @pytest.mark.parametrize("copula", BIVARIATE, ids=_identifier)
    def test_upper_tail_converges_to_the_closed_form(self, copula: rc.Copula) -> None:
        target = copula.lambda_().upper
        sequence = self.upper_sequence(copula, self.RADII)
        assert abs(sequence[-1] - target) < 0.02
        assert abs(sequence[-1] - target) <= abs(sequence[0] - target) + 1e-9

    def test_the_gaussian_copula_has_no_tail_dependence_but_converges_slowly(self) -> None:
        # The classic caution: at rho = 0.9 the ratio is still 0.30 at u = 1e-5,
        # so an empirical estimate at any feasible sample size sees tail
        # dependence that is not there in the limit.
        copula = rc.GaussianCopula(0.9)
        assert copula.lambda_().lower == 0.0
        sequence = self.lower_sequence(copula, np.array([1e-3, 1e-5, 1e-10, 1e-20]))
        assert sequence[0] > 0.35
        assert sequence[1] > 0.25
        assert np.all(np.diff(sequence) < 0)  # decreasing towards zero
        assert sequence[-1] < 0.15

    def test_clayton_lower_and_gumbel_upper(self) -> None:
        for theta in (0.5, 2.0, 6.0):
            assert rc.ClaytonCopula(theta).lambda_().lower == pytest.approx(2.0 ** (-1.0 / theta))
            assert rc.ClaytonCopula(theta).lambda_().upper == 0.0
        for theta in (1.5, 3.0):
            assert rc.GumbelCopula(theta).lambda_().upper == pytest.approx(
                2.0 - 2.0 ** (1.0 / theta)
            )
            assert rc.GumbelCopula(theta).lambda_().lower == 0.0

    def test_student_tail_dependence_formula(self) -> None:
        # lambda = 2 t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho))), symmetric in both
        # tails. Positive for every finite df, at every rho above -1 -- which is
        # the whole reason the t copula replaced the Gaussian one after 2008.
        for rho in (-0.5, 0.0, 0.5, 0.9):
            for df in (2.0, 4.0, 10.0):
                expected = 2.0 * stats.t(df + 1).cdf(
                    -np.sqrt((df + 1.0) * (1.0 - rho) / (1.0 + rho))
                )
                observed = rc.StudentCopula(rho, df=df).lambda_()
                assert observed.lower == pytest.approx(expected, abs=1e-10)
                assert observed.upper == pytest.approx(expected, abs=1e-10)
                assert observed.lower > 0.0


class TestFrechetHoeffdingBounds:
    r""":math:`\max(u+v-1, 0) \le C(u,v) \le \min(u,v)` for every copula.

    Fails loudly for anything that is not a copula, which makes it the cheapest
    possible check on a newly added family or structural construction.
    """

    GRID: ClassVar[np.ndarray] = np.array(
        [[u, v] for u in np.linspace(0.001, 0.999, 40) for v in np.linspace(0.001, 0.999, 40)]
    )

    @pytest.mark.parametrize("copula", BIVARIATE, ids=_identifier)
    def test_the_cdf_lies_between_them(self, copula: rc.Copula) -> None:
        values = np.asarray(copula.cdf(self.GRID))
        lower = np.maximum(self.GRID.sum(axis=1) - 1.0, 0.0)
        upper = np.minimum(self.GRID[:, 0], self.GRID[:, 1])
        assert np.all(values >= lower - 1e-12)
        assert np.all(values <= upper + 1e-12)

    @pytest.mark.parametrize("copula", BIVARIATE, ids=_identifier)
    def test_the_margins_are_uniform(self, copula: rc.Copula) -> None:
        # C(u, 1) = u and C(1, v) = v is what makes it a copula rather than
        # merely a distribution function.
        u = np.linspace(0.01, 0.99, 50)
        np.testing.assert_allclose(copula.cdf(np.column_stack([u, np.ones_like(u)])), u, atol=1e-9)
        np.testing.assert_allclose(copula.cdf(np.column_stack([np.ones_like(u), u])), u, atol=1e-9)

    @pytest.mark.parametrize("copula", BIVARIATE, ids=_identifier)
    def test_a_zero_margin_gives_zero(self, copula: rc.Copula) -> None:
        u = np.linspace(0.01, 0.99, 20)
        np.testing.assert_allclose(
            copula.cdf(np.column_stack([u, np.zeros_like(u)])), 0.0, atol=1e-12
        )

    @pytest.mark.parametrize("copula", BIVARIATE, ids=_identifier)
    def test_the_measures_respect_the_bounds(self, copula: rc.Copula) -> None:
        for value in (copula.tau(), copula.rho(), copula.beta()):
            assert -1.0 <= value <= 1.0
        tails = copula.lambda_()
        assert 0.0 <= tails.lower <= 1.0
        assert 0.0 <= tails.upper <= 1.0


class TestFrailtyCorrespondences:
    """Archimedean copulas are mixtures, and the mixing variable is nameable.

    Each of these says a *sampling construction* -- draw a frailty, divide
    exponentials by it -- produces exactly the copula whose generator is that
    frailty's Laplace transform. It is an identity, not an approximation, so the
    only tolerance is Monte Carlo noise.
    """

    def test_gamma_frailty_gives_clayton(self) -> None:
        # Clayton (1978). psi(t) = (1+t)^(-1/theta) is the Laplace transform of
        # a Gamma(1/theta, 1) variable.
        rng = np.random.default_rng(0)
        for theta in (0.5, 1.0, 2.0, 4.0):
            frailty = rng.gamma(1.0 / theta, scale=theta, size=300_000)
            lifetimes = rng.exponential(1.0, size=(300_000, 2)) / frailty[:, None]
            observed = float(stats.kendalltau(lifetimes[:, 0], lifetimes[:, 1]).statistic)
            assert observed == pytest.approx(rc.ClaytonCopula(theta).tau(), abs=0.006)

    def test_positive_stable_frailty_gives_gumbel(self) -> None:
        # Hougaard (1986). The generator exp(-t^(1/theta)) is the Laplace
        # transform of a positive stable variable with index 1/theta.
        from rcopula.special.stable import rstable_positive

        rng = np.random.default_rng(0)
        for theta in (1.5, 2.0, 3.0):
            frailty = rstable_positive(300_000, 1.0 / theta, rng)
            lifetimes = rng.exponential(1.0, size=(300_000, 2)) / frailty[:, None]
            observed = float(stats.kendalltau(lifetimes[:, 0], lifetimes[:, 1]).statistic)
            assert observed == pytest.approx(rc.GumbelCopula(theta).tau(), abs=0.006)

    def test_the_generator_is_the_laplace_transform_of_the_frailty(self) -> None:
        # The statement underneath both constructions, checked directly:
        # E[exp(-t V)] = psi(t).
        # Scale matters here and nowhere else: the *copula* is invariant to
        # rescaling the frailty, because the scale is absorbed by the
        # exponentials it divides. The generator is not -- psi(t) = (1+t)^(-1/theta)
        # is the transform of Gamma(1/theta, scale = 1) specifically.
        rng = np.random.default_rng(1)
        theta = 2.0
        frailty = rng.gamma(1.0 / theta, scale=1.0, size=2_000_000)
        copula = rc.ClaytonCopula(theta)
        for t in (0.1, 0.5, 1.0, 3.0):
            transform = float(np.mean(np.exp(-t * frailty)))
            assert transform == pytest.approx(
                float(copula.generator.psi(np.array([t]), theta)[0]), abs=0.002
            )


class TestStructuralIdentities:
    """Results that connect copulas to methods usually taught without them."""

    def test_the_nataf_transform_is_a_gaussian_copula(self) -> None:
        # Lebrun and Dutfoy. Structural reliability's standard transformation
        # turns out to be a Gaussian copula, which means it inherits zero tail
        # dependence -- usually unremarked.
        rng = np.random.default_rng(0)
        correlation = 0.7
        margins = [stats.lognorm(0.4, scale=10.0), stats.weibull_min(2.0, scale=8.0)]
        z = rng.multivariate_normal([0, 0], [[1, correlation], [correlation, 1]], size=200_000)
        nataf = np.column_stack([m.ppf(stats.norm.cdf(z[:, j])) for j, m in enumerate(margins)])
        direct = rc.CopulaDistribution(rc.GaussianCopula(correlation), margins).rvs(
            200_000, random_state=0
        )
        assert float(rc.cor_kendall(nataf)[0, 1]) == pytest.approx(
            float(rc.cor_kendall(direct)[0, 1]), abs=0.01
        )
        assert rc.GaussianCopula(correlation).lambda_().upper == 0.0

    def test_the_nonparanormal_covariance_identity(self) -> None:
        # Liu et al.'s SKEPTIC: Sigma = sin(pi tau / 2) recovers the latent
        # correlation whatever the margins were, because tau does not see them.
        rng = np.random.default_rng(0)
        for rho in (-0.6, 0.0, 0.45, 0.85):
            z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=200_000)
            # Two brutal but strictly increasing marginal transforms.
            warped = np.column_stack([np.exp(z[:, 0] / 2), np.sinh(3 * z[:, 1])])
            recovered = float(np.sin(np.pi * rc.cor_kendall(warped)[0, 1] / 2.0))
            assert recovered == pytest.approx(rho, abs=0.01)

    def test_mutual_information_is_negative_copula_entropy(self) -> None:
        # Ma and Sun. For a Gaussian copula the closed form is
        # -1/2 log(1 - rho^2), and the margins contribute nothing at all.
        for rho in (0.3, 0.6, 0.9):
            copula = rc.GaussianCopula(rho)
            u = copula.rvs(400_000, random_state=1)
            information = float(np.mean(copula.logpdf(u)))
            assert information == pytest.approx(-0.5 * np.log(1 - rho**2), abs=0.01)

    def test_rank_measures_are_invariant_under_increasing_transforms(self) -> None:
        # The property that makes copulas worth the trouble. Note the transform
        # must be *increasing*: -log(u) is decreasing and flips the sign.
        copula = rc.ClaytonCopula(2.0)
        u = copula.rvs(20_000, random_state=0)
        increasing = np.column_stack([np.exp(u[:, 0]), np.tan(u[:, 1] * 1.5)])
        assert float(rc.cor_kendall(increasing)[0, 1]) == pytest.approx(
            float(rc.cor_kendall(u)[0, 1]), abs=1e-12
        )
        decreasing = np.column_stack([u[:, 0], -np.log1p(-u[:, 1])])
        assert float(rc.cor_kendall(decreasing)[0, 1]) == pytest.approx(
            float(rc.cor_kendall(u)[0, 1]), abs=1e-12
        )

    def test_survival_of_an_archimedean_reverses_the_tails(self) -> None:
        for theta in (1.5, 3.0):
            gumbel = rc.GumbelCopula(theta)
            survival = rc.survival(gumbel)
            assert survival.lambda_().lower == pytest.approx(gumbel.lambda_().upper)
            assert survival.lambda_().upper == pytest.approx(gumbel.lambda_().lower)
            assert survival.tau() == pytest.approx(gumbel.tau(), abs=1e-10)


class TestHighPrecisionConstants:
    """Two values pinned against ``mpmath`` at 25 digits.

    Neither was taken from any implementation in this package: both were
    computed by integrating the analytic CDF in arbitrary precision, so they are
    an external reference in the same sense a published table would be.
    """

    #: rho_S of Clayton at theta = 2, from 12 * int int C - 3 at 25 digits.
    CLAYTON_TWO_RHO = 0.68223383328065628699

    def test_clayton_at_theta_two(self) -> None:
        assert rc.ClaytonCopula(2.0).rho() == pytest.approx(self.CLAYTON_TWO_RHO, abs=1e-10)

    def test_gumbel_at_theta_two_has_the_same_spearman_rho(self) -> None:
        # Not a coincidence of rounding: verified equal to 20 digits in
        # arbitrary precision. Clayton and Gumbel have different CDFs, different
        # tails and different generators, and their Spearman rho curves cross
        # exactly at tau = 1/2. If either implementation drifts, this breaks.
        assert rc.GumbelCopula(2.0).rho() == pytest.approx(self.CLAYTON_TWO_RHO, abs=1e-10)
        assert rc.ClaytonCopula(2.0).tau() == pytest.approx(0.5, abs=1e-12)
        assert rc.GumbelCopula(2.0).tau() == pytest.approx(0.5, abs=1e-12)

    def test_the_curves_genuinely_cross_there(self) -> None:
        # Equal at tau = 1/2 and unequal either side, so it is a crossing rather
        # than the two families agreeing everywhere.
        below = rc.ClaytonCopula.from_tau(0.45).rho() - rc.GumbelCopula.from_tau(0.45).rho()
        above = rc.ClaytonCopula.from_tau(0.55).rho() - rc.GumbelCopula.from_tau(0.55).rho()
        assert below > 1e-5
        assert above < -1e-5

    @pytest.mark.parametrize(
        ("theta", "expected"),
        [(5.0, 0.45670095816011689683), (-3.0, -0.30724695943072378439)],
    )
    def test_frank_debye_value(self, theta: float, expected: float) -> None:
        # From 1 - 4/theta (1 - D_1(theta)) with the Debye function integrated
        # at 30 digits. Both signs, because the Debye integral runs backwards
        # for negative theta and that is where an implementation slips.
        assert rc.FrankCopula(theta).tau() == pytest.approx(expected, abs=1e-13)
