r"""Debye functions :math:`D_n(x)`.

.. math::

    D_n(x) = \frac{n}{x^n} \int_0^x \frac{t^n}{e^t - 1}\,\mathrm{d}t

``scipy`` does not provide these, yet they are exactly what the Frank copula's
dependence measures are made of:

.. math::

    \tau(\theta)  &= 1 - \frac{4}{\theta}\bigl(1 - D_1(\theta)\bigr) \\
    \rho(\theta)  &= 1 - \frac{12}{\theta}\bigl(D_1(\theta) - D_2(\theta)\bigr)

Implemented from the standard series representations:

    Abramowitz, M. and Stegun, I. A. (1964). *Handbook of Mathematical Functions*,
    §27.1. NIST DLMF §25.12 gives the same expansions.

Two regimes are used, switching at :math:`|x| = 2`:

* **Small** :math:`|x| \le 2` — the Bernoulli series, convergent for
  :math:`|x| < 2\pi`:

  .. math::

      D_n(x) = 1 - \frac{n x}{2(n+1)}
             + n \sum_{k \ge 1} \frac{B_{2k}\, x^{2k}}{(2k+n)\,(2k)!}

  Direct evaluation of the integral form here would cancel catastrophically.

* **Large** :math:`|x| > 2` — expand :math:`(e^t-1)^{-1} = \sum_{k\ge1} e^{-kt}`
  and integrate termwise:

  .. math::

      D_n(x) = \frac{n\,n!}{x^n} \sum_{k \ge 1} \frac{1}{k^{n+1}}
               \left[1 - e^{-kx} \sum_{j=0}^{n} \frac{(kx)^j}{j!}\right]

  The bracket tends to :math:`1`, so the terms decay only like
  :math:`k^{-(n+1)}` and truncating the sum directly would leave a zeta-tail
  error of order :math:`K^{-n}` — about 1% at :math:`K = 60`, :math:`n = 1`.
  Instead the constant part is summed in closed form,
  :math:`\sum_{k\ge1} k^{-(n+1)} = \zeta(n+1)`, leaving

  .. math::

      D_n(x) = \frac{n\,n!}{x^n} \left[ \zeta(n+1)
             - \sum_{k \ge 1} \frac{e^{-kx}}{k^{n+1}}
               \sum_{j=0}^{n} \frac{(kx)^j}{j!} \right]

  whose remaining sum decays geometrically like :math:`e^{-kx}`. This also makes
  the correct asymptote :math:`D_n(x) \to n\,n!\,\zeta(n+1)/x^n` exact.

Negative arguments use the exact reflection
:math:`D_n(-x) = D_n(x) + \frac{n x}{n + 1}`, which follows from the series
(only even powers of :math:`x` appear beyond the linear term).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import bernoulli, factorial, zeta

__all__ = ["debye1", "debye2", "debye_n"]

#: Switch point between the Bernoulli series and the exponential sum.
_SPLIT = 2.0

#: Number of Bernoulli terms. At |x| = 2 the series ratio is (x/2pi)^2 ~ 0.10,
#: so 40 even-order terms take the truncation error far below machine epsilon.
_N_BERNOULLI = 40

#: Number of exponential-sum terms. At x = 2 the k-th term carries e^{-2k};
#: 60 terms give e^{-120}, again far below epsilon.
_N_EXP = 60

# B_0, B_2, B_4, ... — only even-index Bernoulli numbers are needed (the odd
# ones vanish beyond B_1). scipy.special.bernoulli returns B_0..B_m.
_B_EVEN = bernoulli(2 * _N_BERNOULLI)[2::2].copy()
_TWO_K = np.arange(1, _N_BERNOULLI + 1) * 2.0
_B_OVER_FACT = _B_EVEN / factorial(_TWO_K)


def _debye_series(x: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """Bernoulli series; valid and accurate for ``|x| < 2*pi``."""
    # x2k[..., k] = x ** (2(k+1))
    x2k = np.power.outer(x, _TWO_K)
    terms = _B_OVER_FACT / (_TWO_K + n) * x2k
    return 1.0 - n * x / (2.0 * (n + 1.0)) + n * terms.sum(axis=-1)


def _debye_exp(x: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """Exponential sum; accurate for ``x`` bounded away from 0. Requires ``x > 0``.

    The constant part of the series is summed in closed form as ``zeta(n+1)`` so
    that only the geometrically-decaying remainder is truncated.
    """
    k = np.arange(1, _N_EXP + 1, dtype=np.float64)
    kx = np.multiply.outer(x, k)  # (..., k)

    # inner[..., k] = sum_{j=0}^{n} (kx)^j / j!
    j = np.arange(n + 1, dtype=np.float64)
    inner = (np.power.outer(kx, j) / factorial(j)).sum(axis=-1)

    with np.errstate(under="ignore"):
        remainder = (np.exp(-kx) * inner / k ** (n + 1)).sum(axis=-1)

    return n * factorial(n) / x**n * (zeta(n + 1.0) - remainder)


def debye_n(x: ArrayLike, n: int) -> NDArray[np.float64]:
    r"""Evaluate the Debye function :math:`D_n(x)` of order ``n``.

    Parameters
    ----------
    x : array_like
        Real arguments. ``x = 0`` returns ``1`` (the limiting value); negative
        arguments are handled exactly by reflection.
    n : int
        Order, ``n >= 1``.

    Returns
    -------
    ndarray
        :math:`D_n(x)`, elementwise.

    Notes
    -----
    :math:`D_n` is decreasing on :math:`(0, \infty)` with :math:`D_n(0) = 1` and
    :math:`D_n(x) \to n\,n!\,\zeta(n+1) / x^n` as :math:`x \to \infty`.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula.special import debye_n
    >>> float(debye_n(0.0, 1))
    1.0
    >>> float(np.round(debye_n(1.0, 1), 10))
    0.7775046341

    The reflection identity holds exactly:

    >>> x = 3.7
    >>> lhs = float(debye_n(-x, 2))
    >>> rhs = float(debye_n(x, 2)) + 2 * x / 3
    >>> bool(np.isclose(lhs, rhs, rtol=1e-14))
    True
    """
    if n < 1:
        raise ValueError(f"Debye order n must be >= 1, got {n}")

    x = np.asarray(x, dtype=np.float64)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)

    out = np.empty_like(x)

    # Reflect negatives onto the positive axis; correct afterwards.
    neg = x < 0.0
    ax = np.abs(x)

    small = ax <= _SPLIT
    if small.any():
        out[small] = _debye_series(ax[small], n)
    if (~small).any():
        out[~small] = _debye_exp(ax[~small], n)

    out[neg] += n * ax[neg] / (n + 1.0)
    out[np.isnan(x)] = np.nan

    return out[0] if scalar else out


def debye1(x: ArrayLike) -> NDArray[np.float64]:
    r"""First-order Debye function :math:`D_1(x)`.

    Appears in the Frank copula's Kendall tau,
    :math:`\tau(\theta) = 1 - 4\bigl(1 - D_1(\theta)\bigr)/\theta`.

    Examples
    --------
    Frank's tau at ``theta = 5``:

    >>> import numpy as np
    >>> from rcopula.special import debye1
    >>> theta = 5.0
    >>> tau = 1 - 4 * (1 - float(debye1(theta))) / theta
    >>> float(np.round(tau, 10))
    0.4567009582
    """
    return debye_n(x, 1)


def debye2(x: ArrayLike) -> NDArray[np.float64]:
    r"""Second-order Debye function :math:`D_2(x)`.

    Appears in the Frank copula's Spearman rho,
    :math:`\rho(\theta) = 1 - 12\bigl(D_1(\theta) - D_2(\theta)\bigr)/\theta`.

    Examples
    --------
    Frank's rho at ``theta = 5``:

    >>> import numpy as np
    >>> from rcopula.special import debye1, debye2
    >>> theta = 5.0
    >>> rho = 1 - 12 * (float(debye1(theta)) - float(debye2(theta))) / theta
    >>> float(np.round(rho, 10))
    0.6434871081
    """
    return debye_n(x, 2)
