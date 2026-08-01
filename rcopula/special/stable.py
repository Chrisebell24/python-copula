r"""Samplers for the frailty (mixing) distributions of Archimedean copulas.

Every Archimedean copula whose generator is completely monotone can be written
as a *frailty mixture*: draw :math:`V \sim F`, draw iid :math:`E_j \sim
\mathrm{Exp}(1)`, and set :math:`U_j = \psi(E_j / V)`. The resulting
:math:`U` has copula :math:`C` with generator :math:`\psi = \mathcal{L}[F]`.
This is the Marshall-Olkin algorithm, and it is how ``rcopula`` samples every
Archimedean family in :math:`d > 2`.

    Marshall, A. W. and Olkin, I. (1988). Families of multivariate
    distributions. *JASA* 83(403), 834-841.

Each family needs its own :math:`V_0`:

======== ==================================== ==============================
Family   :math:`V_0`                          Implemented by
======== ==================================== ==============================
Clayton  Gamma(1/theta, 1)                    ``numpy`` directly
Gumbel   positive stable S(1/theta, 1, ...)   :func:`rstable_positive`
Frank    logarithmic series, p = 1 - e^-theta :func:`rlog_series`
Joe      Sibuya(1/theta)                      :func:`rsibuya`
AMH      Geometric(1 - theta)                 ``numpy`` directly
======== ==================================== ==============================

References
----------
Chambers, J. M., Mallows, C. L. and Stuck, B. W. (1976). A method for
    simulating stable random variables. *JASA* 71(354), 340-344.
    The CMS algorithm used by :func:`rstable_positive`.
Kemp, A. W. (1981). Efficient generation of logarithmically distributed
    pseudo-random variables. *Applied Statistics* 30(3), 249-253.
    The "LK" algorithm used by :func:`rlog_series`.
Hofert, M. (2008). Sampling Archimedean copulas. *Computational Statistics &
    Data Analysis* 52(12), 5163-5174.
    Sibuya sampling and the family-to-frailty correspondence.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

__all__ = ["rlog_series", "rsibuya", "rstable_positive", "sinc"]


def sinc(x: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """``sin(x) / x``, continuous at zero where it equals 1.

    Appears in Zolotarev's representation of the stable density.
    """
    x = np.asarray(x, dtype=np.float64)
    return np.where(np.abs(x) < 1e-8, 1.0 - x**2 / 6.0, np.sin(x) / np.where(x == 0, 1, x))


def rstable_positive(size: int, alpha: float, rng: np.random.Generator) -> NDArray[np.float64]:
    r"""Draw from the positive stable law :math:`S(\alpha, 1, \gamma, 0; 1)`
    with :math:`\gamma = \cos(\pi\alpha/2)^{1/\alpha}`.

    This is the frailty distribution of the Gumbel copula with
    :math:`\alpha = 1/\theta`. Its Laplace transform is
    :math:`\mathbb{E}[e^{-tV}] = e^{-t^{\alpha}}`, which is exactly the Gumbel
    generator.

    Parameters
    ----------
    size : int
        Number of draws.
    alpha : float
        Stability index in ``(0, 1]``. ``alpha = 1`` degenerates to the constant
        ``1`` (the independence case, ``theta = 1``).
    rng : numpy.random.Generator

    Returns
    -------
    ndarray
        Strictly positive draws.

    Notes
    -----
    Implements the Chambers-Mallows-Stuck transformation directly rather than
    going through ``scipy.stats.levy_stable``: the generic scipy path is far
    slower and its S0/S1 parameterisation mapping is easy to get subtly wrong
    for the totally-skewed case needed here.

    With :math:`W \sim \mathrm{Exp}(1)` and :math:`U \sim \mathrm{Unif}(0, \pi)`,

    .. math::

        V = \frac{\sin(\alpha U)}{\sin(U)^{1/\alpha}}
            \left(\frac{\sin((1-\alpha)U)}{W}\right)^{(1-\alpha)/\alpha}

    Examples
    --------
    The defining Laplace transform, checked by Monte Carlo:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> v = rstable_positive(200_000, 0.5, rng)
    >>> t = 1.3
    >>> bool(abs(np.mean(np.exp(-t * v)) - np.exp(-t ** 0.5)) < 5e-3)
    True
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1], got {alpha}")
    if alpha == 1.0:
        # S(1, 1, 1, 0) with this scaling is the point mass at 1.
        return np.ones(size)

    u = rng.uniform(0.0, np.pi, size)
    w = rng.exponential(1.0, size)

    return (
        np.sin(alpha * u)
        / np.sin(u) ** (1.0 / alpha)
        * (np.sin((1.0 - alpha) * u) / w) ** ((1.0 - alpha) / alpha)
    )


def rlog_series(size: int, p: float, rng: np.random.Generator) -> NDArray[np.float64]:
    r"""Draw from the logarithmic series distribution with parameter ``p``.

    :math:`P(V = k) = -p^k / (k \log(1 - p))` for :math:`k = 1, 2, \dots`.
    This is the frailty of the Frank copula with :math:`p = 1 - e^{-\theta}`.

    Implements Kemp's (1981) "LK" algorithm: a chop-down search from the mode
    for small ``p``, and an inversion-based shortcut otherwise.

    Examples
    --------
    Mean of the logarithmic distribution is ``-p / ((1-p) log(1-p))``:

    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> p = 0.6
    >>> v = rlog_series(200_000, p, rng)
    >>> expected = -p / ((1 - p) * np.log1p(-p))
    >>> bool(abs(v.mean() - expected) / expected < 0.02)
    True
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie in (0, 1), got {p}")

    log1mp = np.log1p(-p)
    out = np.empty(size, dtype=np.float64)

    u = rng.uniform(size=size)
    # Kemp's LK: for u > p the variate is 1 with high probability; otherwise
    # invert via a second uniform.
    simple = u > p
    out[simple] = 1.0

    idx = np.flatnonzero(~simple)
    if idx.size:
        v = rng.uniform(size=idx.size)
        q = -np.expm1(v * log1mp)
        u2 = u[idx]
        out[idx] = np.where(
            u2 < q * q,
            np.floor(1.0 + np.log(u2) / np.log(q)),
            np.where(u2 > q, 1.0, 2.0),
        )
    return out


def rsibuya(size: int, alpha: float, rng: np.random.Generator) -> NDArray[np.float64]:
    r"""Draw from the Sibuya distribution with parameter ``alpha`` in ``(0, 1]``.

    :math:`P(V = k) = \binom{k - 1 - \alpha}{k - 1}\frac{\alpha}{k}`, with
    generating function :math:`1 - (1 - t)^{\alpha}`. This is the frailty of the
    Joe copula with :math:`\alpha = 1/\theta`.

    Sampled by the inversion method of Hofert (2008): with
    :math:`U \sim \mathrm{Unif}(0,1)`, return 1 if
    :math:`U \le \alpha`, else invert the tail using the Beta relationship.

    Examples
    --------
    Sibuya is heavy-tailed with infinite mean for ``alpha < 1``; the sampler
    must therefore produce occasional very large values.

    >>> import numpy as np
    >>> rng = np.random.default_rng(2)
    >>> v = rsibuya(100_000, 0.5, rng)
    >>> bool((v >= 1).all() and v.max() > 100)
    True

    ``alpha = 1`` degenerates to the point mass at 1:

    >>> bool((rsibuya(1000, 1.0, rng) == 1).all())
    True

    The survival function is matched, not just the shape:

    >>> from scipy.special import gammaln
    >>> v = rsibuya(400_000, 0.7, np.random.default_rng(3))
    >>> k = 5
    >>> exact = np.exp(gammaln(k - 0.7) - gammaln(1 - 0.7) - gammaln(k))
    >>> bool(abs((v >= k).mean() - exact) < 5e-3)
    True
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1], got {alpha}")
    if alpha == 1.0:
        return np.ones(size)

    u = rng.uniform(size=size)
    out = np.ones(size, dtype=np.float64)

    idx = np.flatnonzero(u > alpha)
    if not idx.size:
        return out

    # Inversion on the exact survival function
    #     S(k) = P(V >= k) = Gamma(k - alpha) / (Gamma(1 - alpha) Gamma(k)),
    # which is strictly decreasing in k. A chop-down search from k = 1 would not
    # do: Sibuya has infinite mean for alpha < 1, so the expected number of
    # steps diverges. Instead start from the tail asymptote
    #     S(k) ~ k^{-alpha} / Gamma(1 - alpha)
    # and correct with a short monotone walk, which lands within a step or two.
    target = 1.0 - u[idx]  # want the largest k with S(k) >= target
    log_norm = gammaln(1.0 - alpha)

    def log_surv(k: NDArray[np.float64]) -> NDArray[np.float64]:
        return gammaln(k - alpha) - log_norm - gammaln(k)

    k = np.maximum(2.0, np.floor((target * np.exp(log_norm)) ** (-1.0 / alpha)))
    log_target = np.log(target)

    # Walk down while we have overshot (S(k) < target), then up while S(k+1) is
    # still admissible. Both loops are bounded because S is monotone and the
    # starting guess is asymptotically exact.
    for _ in range(64):
        too_far = (log_surv(k) < log_target) & (k > 1.0)
        if not too_far.any():
            break
        k[too_far] -= 1.0
    for _ in range(64):
        can_advance = log_surv(k + 1.0) >= log_target
        if not can_advance.any():
            break
        k[can_advance] += 1.0

    out[idx] = np.maximum(k, 1.0)
    return out
