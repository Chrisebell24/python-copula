r"""Multivariate normal and Student-t distribution functions.

``scipy.stats.multivariate_normal.cdf`` exists but is unsuitable here for two
reasons. It is **not deterministic** — two identical calls on a 10-dimensional
problem differ by ~5e-6 — and it accepts no ``random_state``, so the
non-determinism cannot even be pinned down. A copula library needs a CDF that
returns the same number every time, both for reproducible research and so that
golden fixtures mean anything.

This module therefore provides its own, with the accuracy split by dimension:

* ``d = 2``: **exact**, via Owen's T function. No quadrature at all.
* ``d >= 3``: the **Genz separation-of-variables transformation** evaluated on a
  scrambled Sobol sequence with a fixed seed. Randomised QMC converges far
  faster than plain Monte Carlo on this integrand and, with the seed pinned, is
  reproducible to the last bit.

References
----------
Genz, A. (1992). Numerical computation of multivariate normal probabilities.
    *Journal of Computational and Graphical Statistics* 1(2), 141-149.
Genz, A. and Bretz, F. (2009). *Computation of Multivariate Normal and t
    Probabilities*. Springer Lecture Notes in Statistics 195.
    Chapter 4.2 gives the Student-t radial mixture used in :func:`mvt_cdf`.
Owen, D. B. (1956). Tables for computing bivariate normal probabilities.
    *Annals of Mathematical Statistics* 27(4), 1075-1090.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln, ndtr, ndtri, owens_t
from scipy.stats import chi2, qmc

__all__ = ["bvn_cdf", "mvn_cdf", "mvt_cdf"]

#: Sobol points (2**k) used by the Genz-Bretz integrator.
_QMC_LOG2_POINTS = 16

#: Fixed scramble seed. This is what makes the result reproducible; it must not
#: be exposed as a tunable, or golden fixtures stop being comparable.
_QMC_SEED = 20260731

#: Gauss-Legendre nodes for the Student-t radial integral.
_T_RADIAL_NODES = 256


def bvn_cdf(h: ArrayLike, k: ArrayLike, rho: float) -> NDArray[np.float64]:
    r"""Exact standard bivariate normal CDF :math:`\Phi_2(h, k; \rho)`.

    Uses the Owen (1956) decomposition

    .. math::

        \Phi_2(h,k;\rho) = \tfrac{1}{2}\bigl(\Phi(h) + \Phi(k)\bigr)
                           - T(h, a_h) - T(k, a_k) - \beta

    with :math:`a_h = (k/h - \rho)/\sqrt{1-\rho^2}`, and :math:`\beta = 1/2`
    when :math:`hk < 0` (or :math:`hk = 0` with :math:`h + k < 0`), else 0.

    Parameters
    ----------
    h, k : array_like
        Upper integration limits, broadcast against each other.
    rho : float
        Correlation in ``[-1, 1]``.

    Returns
    -------
    ndarray
        :math:`P(X \le h,\ Y \le k)`.

    Examples
    --------
    Independence factorises:

    >>> import numpy as np
    >>> from rcopula.special.mvtnorm import bvn_cdf
    >>> float(np.round(bvn_cdf(0.5, 0.3, 0.0), 12))
    0.427262552836
    >>> float(np.round(bvn_cdf(0.5, 0.3, 0.0) - ndtr_(0.5) * ndtr_(0.3), 15))
    0.0

    At the origin there is a closed form, ``1/4 + arcsin(rho)/(2*pi)``:

    >>> rho = 0.6
    >>> exact = 0.25 + np.arcsin(rho) / (2 * np.pi)
    >>> bool(np.isclose(bvn_cdf(0.0, 0.0, rho), exact, rtol=1e-14))
    True
    """
    h = np.asarray(h, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    h, k = np.broadcast_arrays(h, k)

    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho must lie in [-1, 1], got {rho}")

    # Degenerate correlations have closed forms and would divide by zero below.
    if rho == 1.0:
        return ndtr(np.minimum(h, k))
    if rho == -1.0:
        return np.maximum(ndtr(h) + ndtr(k) - 1.0, 0.0)

    s = np.sqrt(1.0 - rho * rho)

    # T(0, a) = arctan(a) / (2 pi), and a -> +-inf as the other limit -> 0.
    # Handle h = 0 and k = 0 by substituting the limiting arguments rather than
    # dividing by zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        a_h = (k / h - rho) / s
        a_k = (h / k - rho) / s
    a_h = np.where(h == 0.0, np.where(k >= 0.0, np.inf, -np.inf) / s, a_h)
    a_k = np.where(k == 0.0, np.where(h >= 0.0, np.inf, -np.inf) / s, a_k)
    # Both limits zero gives 0/0. Writing the argument in its unreduced form,
    # a_h = (k - rho*h) / (h*s), and approaching along h = k shows the limit is
    # (1 - rho)/s -- not -rho/s, which would double the answer at the origin.
    both_zero = (h == 0.0) & (k == 0.0)
    a_h = np.where(both_zero, (1.0 - rho) / s, a_h)
    a_k = np.where(both_zero, (1.0 - rho) / s, a_k)

    hk = h * k
    beta = np.where((hk < 0.0) | ((hk == 0.0) & (h + k < 0.0)), 0.5, 0.0)

    return 0.5 * (ndtr(h) + ndtr(k)) - owens_t(h, a_h) - owens_t(k, a_k) - beta


def ndtr_(x: float) -> float:  # pragma: no cover - doctest helper
    """Standard normal CDF, exposed so the doctest above reads cleanly."""
    return float(ndtr(x))


def _genz_transform(
    upper: NDArray[np.float64],
    chol: NDArray[np.float64],
    w: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Genz's separation-of-variables estimator, vectorised over QMC points.

    Recursively conditions on each coordinate: the probability is the product of
    the per-coordinate conditional normal masses, whose arguments depend on the
    previously drawn (transformed) values.
    """
    d = upper.shape[0]
    n = w.shape[0]

    e = ndtr(upper[0] / chol[0, 0])
    prod = np.full(n, e)
    y = np.zeros((n, d))

    for i in range(1, d):
        # Invert within the already-accepted mass, so every point contributes.
        arg = np.clip(w[:, i - 1] * e, 1e-16, 1.0 - 1e-16)
        y[:, i - 1] = ndtri(arg)
        num = upper[i] - y[:, :i] @ chol[i, :i]
        e = ndtr(num / chol[i, i])
        prod *= e

    return prod


def mvn_cdf(
    upper: ArrayLike,
    corr: ArrayLike,
    *,
    n_points: int | None = None,
) -> NDArray[np.float64]:
    r"""Multivariate normal CDF with unit variances.

    Parameters
    ----------
    upper : array_like
        ``(n, d)`` (or ``(d,)``) array of upper integration limits.
    corr : array_like
        ``(d, d)`` correlation matrix.
    n_points : int, optional
        Number of QMC points for ``d >= 3``. Defaults to ``2**14``.

    Returns
    -------
    ndarray
        ``P(X_1 <= upper_1, ..., X_d <= upper_d)`` for each row.

    Notes
    -----
    Deterministic: repeated calls with the same arguments return bit-identical
    results. ``d = 2`` is exact; higher dimensions carry a QMC error of roughly
    1e-8 at the default point count.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula.special.mvtnorm import mvn_cdf
    >>> corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> float(np.round(mvn_cdf([[0.0, 0.0]], corr)[0], 12))
    0.333333333333

    Determinism, which ``scipy``'s version does not offer:

    >>> c = np.eye(5) * 0.5 + 0.5
    >>> a = mvn_cdf([[0.3] * 5], c)
    >>> b = mvn_cdf([[0.3] * 5], c)
    >>> bool(a[0] == b[0])
    True
    """
    x = np.atleast_2d(np.asarray(upper, dtype=np.float64))
    r = np.asarray(corr, dtype=np.float64)
    d = r.shape[0]
    if x.shape[1] != d:
        raise ValueError(f"upper has {x.shape[1]} columns but corr is {d}x{d}")

    if d == 1:
        return ndtr(x[:, 0])
    if d == 2:
        return bvn_cdf(x[:, 0], x[:, 1], float(r[0, 1]))

    n = int(n_points) if n_points else 2**_QMC_LOG2_POINTS
    sampler = qmc.Sobol(d=d - 1, scramble=True, seed=_QMC_SEED)
    w = sampler.random(n)

    out = np.empty(x.shape[0])
    for j, row in enumerate(x):
        # Genz's variable reordering: integrating the most tightly constrained
        # coordinate first makes the inner conditional masses least variable,
        # which is where nearly all the QMC error comes from. Sorting per row is
        # cheap next to the integration itself and buys about two digits.
        order = np.argsort(row)
        chol = np.linalg.cholesky(r[np.ix_(order, order)])
        out[j] = _genz_transform(row[order], chol, w).mean()
    return out


def mvt_cdf(
    upper: ArrayLike,
    corr: ArrayLike,
    df: float,
    *,
    n_points: int | None = None,
) -> NDArray[np.float64]:
    r"""Multivariate Student-t CDF with unit scale.

    Uses the radial mixture representation: if :math:`X \sim t_\nu(0, R)` then
    :math:`X = Z / \sqrt{W/\nu}` with :math:`Z \sim N(0, R)` and
    :math:`W \sim \chi^2_\nu`, so the t probability is the normal probability at
    rescaled limits, averaged over :math:`W` (Genz & Bretz 2009, §4.2).

    Unlike R's ``mvtnorm::pmvt``, **non-integer degrees of freedom are
    supported** — the radial variable is continuous, so there is no reason to
    restrict it, and t copulas are routinely fitted with fractional ``df``.

    Parameters
    ----------
    upper : array_like
        ``(n, d)`` array of upper integration limits.
    corr : array_like
        ``(d, d)`` correlation matrix.
    df : float
        Degrees of freedom, ``> 0``. May be non-integer.
    n_points : int, optional
        Number of QMC points. Defaults to ``2**14``.

    Examples
    --------
    Large ``df`` converges to the normal case:

    >>> import numpy as np
    >>> from rcopula.special.mvtnorm import mvn_cdf, mvt_cdf
    >>> corr = np.array([[1.0, 0.4], [0.4, 1.0]])
    >>> t_big = mvt_cdf([[0.5, -0.2]], corr, df=1e6)[0]
    >>> normal = mvn_cdf([[0.5, -0.2]], corr)[0]
    >>> bool(abs(t_big - normal) < 1e-4)
    True

    Non-integer degrees of freedom, which ``mvtnorm::pmvt`` rejects:

    >>> float(mvt_cdf([[0.0, 0.0]], corr, df=3.5)[0]) > 0.3
    True
    """
    x = np.atleast_2d(np.asarray(upper, dtype=np.float64))
    r = np.asarray(corr, dtype=np.float64)
    d = r.shape[0]
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}")
    if x.shape[1] != d:
        raise ValueError(f"upper has {x.shape[1]} columns but corr is {d}x{d}")

    # Two strategies, because the cost profile differs sharply by dimension.
    #
    # d = 2: integrate the radial variable with Gauss-Legendre against the chi
    # density, calling the *exact* bivariate normal CDF at each node. Each call
    # is a couple of Owen's T evaluations, so 256 nodes is cheap, and the result
    # is exact to ~1e-14.
    #
    # d >= 3: fold the radial variable into the QMC sample as one extra
    # dimension. Reusing the d = 2 approach here would mean 256 separate
    # 65536-point QMC integrations per row -- hundreds of times slower for no
    # accuracy gain, since the inner MVN is itself only good to ~1e-7.
    if d == 2:
        rho = float(r[0, 1])
        s_max = float(np.sqrt(chi2.ppf(1.0 - 1e-15, df) / df))
        nodes, weights = np.polynomial.legendre.leggauss(_T_RADIAL_NODES)
        s_nodes = 0.5 * s_max * (nodes + 1.0)
        s_weights = 0.5 * s_max * weights

        # Density of S = sqrt(W/df), W ~ chi2_df:
        #   f_S(s) = 2 (df/2)^{df/2} / Gamma(df/2) * s^{df-1} exp(-df s^2 / 2)
        half = df / 2.0
        dens = np.exp(
            np.log(2.0)
            + half * np.log(half)
            - gammaln(half)
            + (df - 1.0) * np.log(s_nodes)
            - half * s_nodes**2
        )

        out = np.zeros(x.shape[0])
        for weight, scale, f in zip(s_weights, s_nodes, dens, strict=True):
            if weight * f == 0.0:
                continue
            out += weight * f * bvn_cdf(x[:, 0] * scale, x[:, 1] * scale, rho)
        return out

    n = int(n_points) if n_points else 2**_QMC_LOG2_POINTS
    sampler = qmc.Sobol(d=d, scramble=True, seed=_QMC_SEED)
    w = sampler.random(n)
    # First coordinate carries the radial variable by inversion.
    scale = np.sqrt(chi2.ppf(np.clip(w[:, 0], 1e-16, 1.0 - 1e-16), df) / df)
    rest = w[:, 1:]

    out = np.empty(x.shape[0])
    for j, row in enumerate(x):
        order = np.argsort(row)
        chol = np.linalg.cholesky(r[np.ix_(order, order)])
        out[j] = _genz_transform_scaled(row[order], chol, rest, scale).mean()
    return out


def _genz_transform_scaled(
    upper: NDArray[np.float64],
    chol: NDArray[np.float64],
    w: NDArray[np.float64],
    scale: NDArray[np.float64],
) -> NDArray[np.float64]:
    """As :func:`_genz_transform`, but with per-point rescaled limits.

    The Student-t mixture rescales the integration limits by a draw of the
    radial variable, which differs from point to point; the plain version takes
    a single fixed limit vector.
    """
    d = upper.shape[0]
    n = w.shape[0]

    e = ndtr(upper[0] * scale / chol[0, 0])
    prod = e.copy()
    y = np.zeros((n, d))

    for i in range(1, d):
        arg = np.clip(w[:, i - 1] * e, 1e-16, 1.0 - 1e-16)
        y[:, i - 1] = ndtri(arg)
        num = upper[i] * scale - y[:, :i] @ chol[i, :i]
        e = ndtr(num / chol[i, i])
        prod *= e

    return prod
