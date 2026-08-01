"""Accurate evaluation of ``log(1 - exp(-a))`` and ``log(1 + exp(x))``.

Both expressions are numerically treacherous: each has a regime where the naive
formula cancels catastrophically and a regime where it overflows. The
piecewise-optimal cutoffs used here are derived in

    Mächler, M. (2012). *Accurately Computing log(1 - exp(-|a|))*.
    Assessed by the Rmpfr package vignette.
    https://cran.r-project.org/package=Rmpfr/vignettes/log1mexp-note.pdf

For ``log1mexp`` the optimal switch is at ``a = log 2``: below it ``log(-expm1(-a))``
is accurate, above it ``log1p(-exp(-a))`` is. For ``log1pexp`` the note gives the
four-regime cutoffs -37 / 18 / 33.3, chosen so that the discarded term falls below
the double-precision epsilon.

Also provided is a sign-aware log-sum-exp, needed because the ``d``-dimensional
Archimedean densities are alternating sums whose terms span many orders of
magnitude.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["log1mexp", "log1pexp", "signed_logsumexp"]

_LOG2 = float(np.log(2.0))


def log1mexp(a: ArrayLike) -> NDArray[np.float64]:
    """Compute ``log(1 - exp(-a))`` accurately for ``a > 0``.

    Parameters
    ----------
    a : array_like
        Non-negative values. ``a = 0`` yields ``-inf`` (since ``log 0 = -inf``);
        negative values yield ``nan``, because ``1 - exp(-a) < 0`` there.

    Returns
    -------
    ndarray
        ``log(1 - exp(-a))``, elementwise.

    Notes
    -----
    Uses ``log(-expm1(-a))`` for ``a <= log 2`` and ``log1p(-exp(-a))`` above it,
    the switch point that minimises the worst-case relative error (Mächler 2012).

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula.special import log1mexp
    >>> float(log1mexp(1e-10))          # naive log(1 - exp(-a)) loses all precision here
    -23.025850929990458
    >>> bool(np.isinf(log1mexp(0.0)))
    True
    """
    a = np.asarray(a, dtype=np.float64)
    out = np.full(a.shape, np.nan, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        small = a <= _LOG2
        # Only evaluate where the branch applies, so the other branch cannot
        # generate spurious warnings on out-of-range inputs.
        lo = small & (a >= 0.0)
        hi = ~small & np.isfinite(a) | (~small & np.isinf(a))
        out[lo] = np.log(-np.expm1(-a[lo]))
        out[hi] = np.log1p(-np.exp(-a[hi]))

    return out[()] if out.ndim == 0 else out


def log1pexp(x: ArrayLike) -> NDArray[np.float64]:
    """Compute ``log(1 + exp(x))`` accurately for all real ``x``.

    Parameters
    ----------
    x : array_like
        Any real values.

    Returns
    -------
    ndarray
        ``log(1 + exp(x))``, elementwise. This is the softplus function.

    Notes
    -----
    Four regimes, with cutoffs from Mächler (2012):

    ==================  ==============================
    Range               Formula
    ==================  ==============================
    ``x <= -37``        ``exp(x)``
    ``-37 < x <= 18``   ``log1p(exp(x))``
    ``18 < x <= 33.3``  ``x + exp(-x)``
    ``x > 33.3``        ``x``
    ==================  ==============================

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula.special import log1pexp
    >>> float(log1pexp(0.0))
    0.6931471805599453
    >>> float(log1pexp(1000.0))         # naive exp(1000) would overflow
    1000.0
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty(x.shape, dtype=np.float64)

    with np.errstate(over="ignore"):
        r1 = x <= -37.0
        r2 = (x > -37.0) & (x <= 18.0)
        r3 = (x > 18.0) & (x <= 33.3)
        r4 = x > 33.3

        out[r1] = np.exp(x[r1])
        out[r2] = np.log1p(np.exp(x[r2]))
        out[r3] = x[r3] + np.exp(-x[r3])
        out[r4] = x[r4]

    # NaN in, NaN out (none of the comparisons above select NaN).
    out[np.isnan(x)] = np.nan
    return out[()] if out.ndim == 0 else out


def signed_logsumexp(
    log_abs: ArrayLike,
    signs: ArrayLike,
    axis: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sum signed terms given in log-absolute form, without leaving log space.

    Computes ``log|S|`` and ``sign(S)`` where ``S = sum(signs * exp(log_abs))``.

    This is the workhorse for ``d``-dimensional Archimedean densities, whose
    generator derivatives are alternating sums with terms spanning hundreds of
    orders of magnitude: forming them directly overflows, and dropping the sign
    information gives the wrong answer.

    Parameters
    ----------
    log_abs : array_like
        Logarithms of the absolute values of the terms.
    signs : array_like
        Signs (``+1`` or ``-1``) of the terms; broadcast against ``log_abs``.
    axis : int, optional
        Axis to sum over. ``None`` sums the flattened input.

    Returns
    -------
    log_abs_sum : ndarray
        ``log|S|``. Equals ``-inf`` where the terms cancel exactly.
    sign_sum : ndarray
        ``sign(S)``, one of ``-1.0``, ``0.0``, ``1.0``.

    Examples
    --------
    Cancellation that would be lost in ordinary arithmetic:

    >>> import numpy as np
    >>> from rcopula.special import signed_logsumexp
    >>> lv, sg = signed_logsumexp([0.0, 0.0], [1.0, -1.0])
    >>> bool(np.isneginf(lv)), float(sg)
    (True, 0.0)

    A plain sum, ``exp(1) - exp(0) = 1.718...``:

    >>> lv, sg = signed_logsumexp([1.0, 0.0], [1.0, -1.0])
    >>> float(sg) * float(np.exp(lv))
    1.7182818284590453
    """
    log_abs = np.asarray(log_abs, dtype=np.float64)
    signs = np.broadcast_to(np.asarray(signs, dtype=np.float64), log_abs.shape)

    # Shift by the running maximum so the largest term is exp(0) = 1.
    m = np.max(np.where(signs != 0.0, log_abs, -np.inf), axis=axis, keepdims=True)
    m_finite = np.where(np.isfinite(m), m, 0.0)

    with np.errstate(invalid="ignore"):
        total = np.sum(signs * np.exp(log_abs - m_finite), axis=axis, keepdims=True)

    sign_sum = np.sign(total)
    with np.errstate(divide="ignore"):
        log_abs_sum = np.squeeze(m_finite + np.log(np.abs(total)), axis=axis)
        sign_sum = np.squeeze(sign_sum, axis=axis)

    # Everything cancelled, or every term was zero.
    log_abs_sum = np.where(sign_sum == 0.0, -np.inf, log_abs_sum)

    return log_abs_sum[()] if log_abs_sum.ndim == 0 else log_abs_sum, (
        sign_sum[()] if sign_sum.ndim == 0 else sign_sum
    )
