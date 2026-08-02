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

from rcopula.special.logexp import log1mexp

__all__ = ["retstable", "rlog_series", "rsibuya", "rstable_positive", "sinc"]


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


def rlog_series(
    size: int, p: float, rng: np.random.Generator, log1mp: float | None = None
) -> NDArray[np.float64]:
    r"""Draw from the logarithmic series distribution with parameter ``p``.

    :math:`P(V = k) = -p^k / (k \log(1 - p))` for :math:`k = 1, 2, \dots`.
    This is the frailty of the Frank copula with :math:`p = 1 - e^{-\theta}`.

    Implements Kemp's (1981) "LK" algorithm: a chop-down search from the mode
    for small ``p``, and an inversion-based shortcut otherwise.

    Parameters
    ----------
    size : int
        Number of draws.
    p : float
        Series parameter in ``(0, 1]``.
    rng : Generator
        Source of randomness.
    log1mp : float, optional
        A separately computed :math:`\log(1 - p)`. Supply it whenever ``p`` was
        formed as ``1 - e^{-\theta}``: past ``theta`` around 37 that expression
        rounds to exactly 1, so ``log1p(-p)`` becomes ``-inf`` and the sampler
        fails outright -- even though the quantity it actually needs is just
        ``-theta``, known exactly. The Frank copula reaches that at
        ``tau = 0.92``, well inside the range people fit.

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

    Passing ``log1mp`` keeps it working where ``p`` itself has saturated:

    >>> theta = 60.0
    >>> p = -np.expm1(-theta)
    >>> bool(p == 1.0)
    True
    >>> v = rlog_series(1000, p, rng, log1mp=-theta)
    >>> bool(np.all(v >= 1))
    True
    """
    if log1mp is None:
        # p == 1 makes this -inf, which the check below reports; computing it
        # should not also raise a warning on the way there.
        with np.errstate(divide="ignore"):
            log1mp = float(np.log1p(-p))
    else:
        log1mp = float(log1mp)
    if not 0.0 < p <= 1.0:
        raise ValueError(f"p must lie in (0, 1], got {p}")
    if not np.isfinite(log1mp) or log1mp >= 0.0:
        raise ValueError(
            f"log(1 - p) must be finite and negative, got {log1mp}; "
            "pass log1mp explicitly when p has rounded to 1"
        )

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
        # log(q) must come from log1mexp, not from log(q). Once
        # ``v * log1mp < -37`` the quantity ``q = 1 - e^{v log1mp}`` rounds to
        # exactly 1, so ``log(q)`` is exactly 0 and the inversion below divides
        # by zero, returning inf. For the Frank copula at theta = 50 that hit a
        # quarter of all draws and sent them to the boundary of the unit square;
        # at theta = 100, nearly two thirds. log1mexp keeps the small negative
        # value the inversion actually needs.
        log_q = log1mexp(-v * log1mp)
        u2 = u[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            inverted = np.floor(1.0 + np.log(u2) / log_q)
        out[idx] = np.where(u2 < q * q, inverted, np.where(u2 > q, 1.0, 2.0))
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


#: Attempts allowed per element before :func:`retstable` gives up. Each round
#: accepts with probability at least ``e^-1``, so exceeding this many is not a
#: slow case but a broken one.
_RETSTABLE_MAX_ROUNDS = 100_000


def retstable(
    size: int,
    alpha: float,
    v0: NDArray[np.float64] | float,
    h: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    r"""Draw from the **exponentially tilted** positive stable distribution.

    Defined by its Laplace transform,

    .. math::
        \mathbb{E}\bigl[e^{-tV}\bigr]
            = \exp\bigl(-V_0\,\bigl[(t + h)^{\alpha} - h^{\alpha}\bigr]\bigr),

    which is the inner frailty of a **nested Clayton** copula and the reason
    nested Archimedean copulas have never been available in Python: this
    distribution has no closed-form density, no quantile function, and no
    implementation in NumPy, SciPy or any copula package.

    Parameters
    ----------
    size : int
        Number of draws. Ignored when ``v0`` is an array, whose length wins.
    alpha : float
        Stability index in ``(0, 1]``. At ``alpha = 1`` the law is degenerate at
        ``v0``.
    v0 : float or ndarray
        The multiplier in the exponent. May vary per draw, which is what the
        nested sampler needs -- each observation carries its own outer frailty.
    h : float
        Tilt. ``h = 0`` gives the untilted stable.
    rng : Generator

    Notes
    -----
    **Method.** Rejection from the untilted stable accepts with probability
    exactly :math:`e^{-V_0 h^\alpha}`, which is fine for a small exponent and
    hopeless otherwise -- worse, for ``V0 ~ Gamma(k, 1)`` with ``k >= 1`` the
    *expected* number of attempts is infinite, so a naive implementation does
    not merely run slowly, it occasionally does not finish.

    Infinite divisibility fixes it. The law is the sum of ``k`` independent
    draws with ``V0/k`` in place of ``V0`` -- the Laplace transforms multiply to
    give back the original -- and each of those accepts with probability
    :math:`e^{-V_0 h^\alpha / k}`. Taking :math:`k = \lceil V_0 h^\alpha\rceil`
    puts every piece's acceptance at about :math:`e^{-1}`, so the work becomes
    **linear** in :math:`V_0 h^\alpha` rather than exponential. The result is
    exact: no approximation enters at any point.

    Examples
    --------
    The defining Laplace transform, checked by Monte Carlo:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> alpha, v0, h, t = 0.5, 2.0, 1.0, 0.8
    >>> v = retstable(200_000, alpha, v0, h, rng)
    >>> want = np.exp(-v0 * ((t + h) ** alpha - h ** alpha))
    >>> bool(abs(np.mean(np.exp(-t * v)) - want) < 3e-3)
    True

    The mean is ``alpha * v0 * h**(alpha - 1)``, by differentiating it:

    >>> bool(abs(v.mean() - alpha * v0 * h ** (alpha - 1)) < 0.02)
    True

    A zero tilt gives the ordinary positive stable back:

    >>> untilted = retstable(5, 0.5, 1.0, 0.0, np.random.default_rng(1))
    >>> bool(np.all(untilted > 0))
    True
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1], got {alpha}")
    if h < 0.0:
        raise ValueError(f"h must be non-negative, got {h}")

    values = np.asarray(v0, dtype=np.float64)
    values = np.full(int(size), float(values)) if values.ndim == 0 else values.ravel()
    if np.any(values < 0.0):
        raise ValueError("v0 must be non-negative")

    if alpha == 1.0:
        # The exponent collapses to V0 * t, a point mass.
        return values.copy()
    if h == 0.0:
        return values ** (1.0 / alpha) * rstable_positive(values.size, alpha, rng)

    exponent = values * h**alpha
    pieces = np.maximum(1, np.ceil(exponent)).astype(np.int64)
    per_piece = values / pieces
    scale = per_piece ** (1.0 / alpha)
    tilt = scale * h  # so each piece accepts with probability about e^-1

    total = np.zeros(values.size)
    outstanding = pieces.copy()
    for _ in range(_RETSTABLE_MAX_ROUNDS):
        active = np.flatnonzero(outstanding > 0)
        if active.size == 0:
            return total
        pending = active
        while pending.size:
            draw = rstable_positive(pending.size, alpha, rng)
            with np.errstate(under="ignore"):
                accepted = rng.uniform(size=pending.size) < np.exp(-tilt[pending] * draw)
            total[pending[accepted]] += scale[pending[accepted]] * draw[accepted]
            pending = pending[~accepted]
        outstanding[active] -= 1

    raise RuntimeError(  # pragma: no cover - unreachable for valid parameters
        "retstable failed to converge; each round accepts with probability at "
        "least exp(-1), so this indicates a bug rather than a slow case"
    )
