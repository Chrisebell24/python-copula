"""Accuracy tests for the Debye functions.

Validated against ``mpmath`` quadrature at 40 digits. Coverage deliberately
straddles the ``|x| = 2`` branch switch and reaches into both tails, since a
truncated series is exactly the kind of bug that hides in the middle of a range
and only shows up far out.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

from rcopula.special import debye1, debye2, debye_n
from rcopula.special.debye import _debye_exp, _debye_series

mp.mp.dps = 40


def _ref_debye(x: float, n: int) -> float:
    """High-precision reference: n/x^n * int_0^x t^n/(e^t - 1) dt."""
    xm = mp.mpf(x)
    if xm == 0:
        return 1.0
    return float(n / xm**n * mp.quad(lambda t: t**n / mp.expm1(t), [mp.mpf("1e-30"), xm]))


# Straddles the branch switch at |x| = 2 and both tails.
ARGS = [
    1e-8,
    1e-4,
    0.01,
    0.1,
    0.5,
    1.0,
    1.9,
    1.999,
    2.0,
    2.001,
    2.1,
    3.0,
    5.0,
    10.0,
    30.0,
    100.0,
    700.0,
    -0.5,
    -1.0,
    -2.5,
    -5.0,
    -50.0,
]


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("x", ARGS)
def test_debye_matches_mpmath(x: float, n: int) -> None:
    assert float(debye_n(x, n)) == pytest.approx(_ref_debye(x, n), rel=1e-13)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_branches_agree_where_they_meet(n: int) -> None:
    """Both series must give the same answer at the same point near the switch.

    Comparing ``D(2-eps)`` against ``D(2+eps)`` would only measure the function's
    own slope, so evaluate the two implementations at identical arguments instead.
    Each is applied slightly outside its designated region, where both are still
    valid: the Bernoulli series converges up to ``2*pi`` and the exponential sum
    is fine down to ``x ~ 1``.
    """
    x = np.array([1.5, 1.9, 2.0, 2.1, 2.5, 3.0])
    assert np.allclose(_debye_series(x, n), _debye_exp(x, n), rtol=1e-13)


def test_limit_at_zero() -> None:
    for n in (1, 2, 3):
        assert float(debye_n(0.0, n)) == pytest.approx(1.0, rel=1e-15)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_large_x_asymptote(n: int) -> None:
    """D_n(x) -> n * n! * zeta(n+1) / x^n."""
    x = 1e6
    expected = n * float(mp.factorial(n)) * float(mp.zeta(n + 1)) / x**n
    assert float(debye_n(x, n)) == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_reflection_identity(n: int) -> None:
    """D_n(-x) = D_n(x) + n*x/(n+1), exactly."""
    x = np.array([0.3, 1.0, 2.5, 7.0, 40.0])
    assert np.allclose(debye_n(-x, n), debye_n(x, n) + n * x / (n + 1.0), rtol=1e-14)


def test_monotone_decreasing_on_positives() -> None:
    x = np.linspace(1e-6, 60, 4001)
    for n in (1, 2, 3):
        assert np.all(np.diff(debye_n(x, n)) < 0)


def test_bounded_in_unit_interval_on_positives() -> None:
    x = np.linspace(1e-9, 500, 2001)
    for n in (1, 2, 3):
        y = debye_n(x, n)
        assert np.all((y > 0) & (y <= 1))


def test_order_must_be_positive() -> None:
    with pytest.raises(ValueError, match="order n must be >= 1"):
        debye_n(1.0, 0)


def test_shape_preservation() -> None:
    assert np.shape(debye1(1.0)) == ()
    assert debye1(np.ones((3, 4))).shape == (3, 4)
    assert debye2(np.ones(7)).shape == (7,)


def test_nan_propagates() -> None:
    assert np.isnan(debye1(np.nan))


class TestFrankDependenceMeasures:
    """The reason the Debye functions are here at all.

    tau(theta) = 1 - 4(1 - D1(theta))/theta
    rho(theta) = 1 - 12(D1(theta) - D2(theta))/theta
    """

    @staticmethod
    def _tau(theta: float) -> float:
        return 1.0 - 4.0 * (1.0 - float(debye1(theta))) / theta

    @staticmethod
    def _rho(theta: float) -> float:
        return 1.0 - 12.0 * (float(debye1(theta)) - float(debye2(theta))) / theta

    @pytest.mark.parametrize("theta", [-40.0, -10.0, -3.0, -0.5, 0.5, 3.0, 10.0, 40.0])
    def test_tau_and_rho_in_range(self, theta: float) -> None:
        assert -1.0 <= self._tau(theta) <= 1.0
        assert -1.0 <= self._rho(theta) <= 1.0

    def test_independence_limit(self) -> None:
        """theta -> 0 is the independence copula: tau = rho = 0."""
        assert self._tau(1e-6) == pytest.approx(0.0, abs=1e-6)
        assert self._rho(1e-6) == pytest.approx(0.0, abs=1e-6)

    def test_antisymmetry_in_theta(self) -> None:
        """Frank is radially symmetric: tau(-theta) = -tau(theta)."""
        for theta in (0.5, 2.0, 7.0, 25.0):
            assert self._tau(-theta) == pytest.approx(-self._tau(theta), rel=1e-12)
            assert self._rho(-theta) == pytest.approx(-self._rho(theta), rel=1e-12)

    def test_tau_is_increasing(self) -> None:
        theta = np.linspace(-50, 50, 501)
        theta = theta[np.abs(theta) > 1e-3]
        assert np.all(np.diff([self._tau(t) for t in theta]) > 0)

    def test_comonotone_limit(self) -> None:
        assert self._tau(1e4) == pytest.approx(1.0, abs=1e-3)
        assert self._rho(1e4) == pytest.approx(1.0, abs=1e-3)

    def test_rho_exceeds_tau_for_positive_dependence(self) -> None:
        """A general fact for Frank: rho >= tau when theta > 0."""
        for theta in (0.5, 1.0, 5.0, 20.0):
            assert self._rho(theta) >= self._tau(theta)
