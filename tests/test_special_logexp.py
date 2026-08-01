"""Accuracy tests for log1mexp / log1pexp / signed_logsumexp.

These are validated against ``mpmath`` at 50 decimal digits, deliberately including
the arguments where the naive formulas fail: near zero for ``log1mexp`` (where
``1 - exp(-a)`` cancels) and in the overflow/underflow tails for ``log1pexp``.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

from rcopula.special import log1mexp, log1pexp, signed_logsumexp


def _dps_for(magnitude: float) -> int:
    """Digits needed to resolve ``1 +/- exp(-|magnitude|)``.

    At ``a = 700`` the true ``log1mexp`` value is ~-9.9e-305, so representing
    ``1 - exp(-a)`` takes over 300 significant digits. Scaling the precision with
    the argument keeps the far tail exact without making every other case slow.

    Capped at 400 digits: past that the correction term is below 1e-400 and
    cannot affect a float64 result, whereas the uncapped formula would ask for
    434,000 digits at ``x = 1e6`` and take over a minute.
    """
    return min(400, int(abs(magnitude) / np.log(10.0)) + 30)


def _ref_log1mexp(a: float) -> float:
    """Reference must use ``expm1``: ``1 - mp.exp(-a)`` loses the very precision
    that ``log1mexp`` exists to preserve."""
    with mp.workdps(_dps_for(a)):
        return float(mp.log(-mp.expm1(-mp.mpf(a))))


def _ref_log1pexp(x: float) -> float:
    """Reference must use ``mp.exp``, not ``mp.e**x`` — the latter is only
    accurate to the precision of the cached constant."""
    with mp.workdps(_dps_for(x)):
        return float(mp.log1p(mp.exp(mp.mpf(x))))


# Spans every branch: deep cancellation region, the log(2) switch, and the tail.
LOG1MEXP_ARGS = [
    1e-300,
    1e-100,
    1e-20,
    1e-16,
    1e-10,
    1e-8,
    1e-4,
    0.1,
    0.5,
    float(np.log(2)) - 1e-12,
    float(np.log(2)),
    float(np.log(2)) + 1e-12,
    1.0,
    2.0,
    10.0,
    37.0,
    100.0,
    500.0,
    700.0,
]

# Spans all four branches of log1pexp, including both cutoffs.
LOG1PEXP_ARGS = [
    -800.0,
    -100.0,
    -37.5,
    -37.0,
    -36.5,
    -1.0,
    0.0,
    1.0,
    17.5,
    18.0,
    18.5,
    33.0,
    33.3,
    33.5,
    100.0,
    700.0,
    1e6,
]


@pytest.mark.parametrize("a", LOG1MEXP_ARGS)
def test_log1mexp_matches_mpmath(a: float) -> None:
    got = float(log1mexp(a))
    expected = _ref_log1mexp(a)
    assert got == pytest.approx(expected, rel=1e-14, abs=1e-300)


@pytest.mark.parametrize("x", LOG1PEXP_ARGS)
def test_log1pexp_matches_mpmath(x: float) -> None:
    got = float(log1pexp(x))
    expected = _ref_log1pexp(x)
    assert got == pytest.approx(expected, rel=1e-14, abs=1e-300)


def test_log1mexp_beats_naive_near_zero() -> None:
    """The whole reason this function exists: naive evaluation loses everything."""
    a = 1e-17
    with np.errstate(divide="ignore"):
        naive = np.log(1.0 - np.exp(-a))  # 1 - exp(-a) rounds to 0.0 -> -inf
    assert np.isinf(naive)
    assert float(log1mexp(a)) == pytest.approx(_ref_log1mexp(a), rel=1e-13)


def test_log1pexp_beats_naive_in_tail() -> None:
    """Naive log(1 + exp(x)) overflows; log1pexp does not."""
    x = 800.0
    with np.errstate(over="ignore"):
        naive = np.log(1.0 + np.exp(x))
    assert np.isinf(naive)
    assert float(log1pexp(x)) == pytest.approx(x, rel=1e-15)


def test_log1mexp_boundaries() -> None:
    assert np.isneginf(log1mexp(0.0))
    assert np.isnan(log1mexp(-1.0))
    assert float(log1mexp(np.inf)) == pytest.approx(0.0, abs=1e-300)


def test_log1pexp_is_monotone_and_nonnegative() -> None:
    x = np.linspace(-50, 50, 2001)
    y = log1pexp(x)
    assert np.all(np.diff(y) >= 0)
    assert np.all(y >= 0)
    # softplus(x) - softplus(-x) == x
    assert np.allclose(log1pexp(x) - log1pexp(-x), x, rtol=1e-12, atol=1e-12)


def test_vectorisation_matches_scalar_calls() -> None:
    a = np.array(LOG1MEXP_ARGS)
    assert np.allclose(log1mexp(a), [float(log1mexp(v)) for v in a], rtol=0, atol=0)
    x = np.array(LOG1PEXP_ARGS)
    assert np.allclose(log1pexp(x), [float(log1pexp(v)) for v in x], rtol=0, atol=0)


def test_shape_preservation() -> None:
    assert np.shape(log1mexp(1.0)) == ()
    assert log1mexp(np.ones((3, 4))).shape == (3, 4)
    assert log1pexp(np.ones((2, 5, 3))).shape == (2, 5, 3)


class TestSignedLogSumExp:
    def test_exact_cancellation(self) -> None:
        lv, sg = signed_logsumexp([0.0, 0.0], [1.0, -1.0])
        assert np.isneginf(lv)
        assert sg == 0.0

    def test_simple_difference(self) -> None:
        lv, sg = signed_logsumexp([1.0, 0.0], [1.0, -1.0])
        assert sg == 1.0
        assert float(np.exp(lv)) == pytest.approx(np.e - 1.0, rel=1e-14)

    def test_negative_result(self) -> None:
        lv, sg = signed_logsumexp([0.0, 1.0], [1.0, -1.0])
        assert sg == -1.0
        assert float(np.exp(lv)) == pytest.approx(np.e - 1.0, rel=1e-14)

    def test_survives_extreme_magnitudes(self) -> None:
        """Terms spanning 600 orders of magnitude, as in d-dimensional AC densities."""
        log_abs = np.array([700.0, 699.0, 100.0, -700.0])
        signs = np.array([1.0, -1.0, 1.0, -1.0])
        lv, sg = signed_logsumexp(log_abs, signs)

        ref = sum(mp.mpf(s) * mp.e ** mp.mpf(la) for la, s in zip(log_abs, signs, strict=True))
        assert sg == float(mp.sign(ref))
        assert float(lv) == pytest.approx(float(mp.log(abs(ref))), rel=1e-13)

    def test_axis_reduction(self) -> None:
        log_abs = np.array([[1.0, 0.0], [2.0, 1.0]])
        signs = np.array([[1.0, -1.0], [1.0, -1.0]])
        lv, sg = signed_logsumexp(log_abs, signs, axis=1)
        assert lv.shape == (2,)
        assert np.allclose(sg, [1.0, 1.0])
        assert np.allclose(np.exp(lv), [np.e - 1.0, np.e**2 - np.e], rtol=1e-14)

    def test_matches_direct_sum_when_safe(self) -> None:
        rng = np.random.default_rng(0)
        log_abs = rng.uniform(-5, 5, size=50)
        signs = rng.choice([-1.0, 1.0], size=50)
        lv, sg = signed_logsumexp(log_abs, signs)
        direct = float(np.sum(signs * np.exp(log_abs)))
        assert sg * float(np.exp(lv)) == pytest.approx(direct, rel=1e-12)
