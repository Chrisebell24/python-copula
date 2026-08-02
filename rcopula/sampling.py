r"""Quasi-random and variance-reduced copula sampling.

Every Monte Carlo answer in this package -- a CDO tranche spread, a basket
option, a portfolio's expected shortfall -- has a standard error, and the
default way to halve it is to quadruple the sample. These are the cheaper ways.

**Quasi-random.** A Sobol or Halton point set fills the unit cube more evenly
than independent uniforms do, because it is *designed* to rather than merely
tending to. Pushing such a set through the inverse Rosenblatt transform gives a
sample from the copula whose integration error falls like :math:`(\log n)^d / n`
rather than :math:`n^{-1/2}`. For a smooth payoff in a few dimensions that is
the difference between four digits and two.

The catch, stated plainly because it is usually not: the improvement depends on
the integrand being smooth and of low effective dimension. A payoff with a kink
(any option), a high dimension, or a discontinuity (a default indicator) erodes
it, and the rate can fall back to :math:`n^{-1/2}`. :func:`variance_ratio`
measures what you actually got instead of assuming it.

**Antithetic.** Pair every draw :math:`z` with :math:`1 - z`. Free, and helps
exactly when the integrand is monotone in the uniforms -- which a European call
on a single asset is, and a straddle is not. It can *increase* variance for a
symmetric payoff, so it is not a default.

**Latin hypercube.** Stratifies each margin into :math:`n` equal bins and
permutes. Guarantees marginal coverage, does nothing about the joint structure,
and is a good default when the answer depends mostly on the margins.

============================  ================================================
:func:`quasi_rvs`             Sobol or Halton draws from a copula.
:func:`antithetic_rvs`        Draws paired with their reflections.
:func:`latin_hypercube_rvs`   Stratified draws.
:func:`variance_ratio`        What the variance reduction actually was.
============================  ================================================

Examples
--------
>>> import numpy as np, rcopula as rc
>>> from rcopula.sampling import quasi_rvs
>>> u = quasi_rvs(rc.ClaytonCopula(2.0), 1024, random_state=0)
>>> u.shape
(1024, 2)
>>> bool(abs(rc.cor_kendall(u)[0, 1] - 0.5) < 0.02)
True

References
----------
Cambou, M., Hofert, M. and Lemieux, C. (2017). Quasi-random numbers for copula
    models. *Statistics and Computing* 27(5), 1307-1329.
    The paper this implements: low-discrepancy sequences through the inverse
    Rosenblatt transform.
Sobol', I. M. (1967). On the distribution of points in a cube and the
    approximate evaluation of integrals.
    *USSR Computational Mathematics and Mathematical Physics* 7(4), 86-112.
Owen, A. B. (1997). Scrambled net variance for integrals of smooth functions.
    *Annals of Statistics* 25(4), 1541-1562.
    Why scrambling is what makes an error estimate possible at all.
McKay, M. D., Beckman, R. J. and Conover, W. J. (1979). A comparison of three
    methods for selecting values of input variables in the analysis of output
    from a computer code. *Technometrics* 21(2), 239-245.
    Latin hypercube sampling.
Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*.
    Springer. Chapter 4 on variance reduction, chapter 5 on quasi-Monte Carlo.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import qmc

from rcopula.core.base import Copula
from rcopula.transforms import inverse_rosenblatt

__all__ = ["antithetic_rvs", "latin_hypercube_rvs", "quasi_rvs", "variance_ratio"]

Sequence = Literal["sobol", "halton"]


def _point_set(
    kind: Sequence, dim: int, size: int, random_state: Any, scramble: bool
) -> NDArray[np.float64]:
    seed = np.random.default_rng(random_state)
    engine: qmc.QMCEngine
    if kind == "sobol":
        engine = qmc.Sobol(d=dim, scramble=scramble, seed=seed)
    elif kind == "halton":
        engine = qmc.Halton(d=dim, scramble=scramble, seed=seed)
    else:
        raise ValueError(f"sequence must be 'sobol' or 'halton', got {kind!r}")
    # Values of exactly 0 or 1 appear in an unscrambled sequence and have no
    # finite quantile, so the cube is opened slightly.
    return np.clip(engine.random(size), 1e-12, 1.0 - 1e-12)


def quasi_rvs(
    copula: Copula,
    size: int,
    *,
    sequence: Sequence = "sobol",
    scramble: bool = True,
    random_state: Any = None,
) -> NDArray[np.float64]:
    r"""Draw from a copula using a low-discrepancy point set.

    Generates a Sobol or Halton set in :math:`[0,1]^d` and pushes it through the
    copula's inverse Rosenblatt transform (Cambou, Hofert and Lemieux 2017). The
    result has the right distribution and covers the space more evenly than
    independent draws.

    Parameters
    ----------
    copula : Copula
    size : int
        Number of points. For Sobol, a power of two preserves the balance
        properties the construction is built on; scipy warns otherwise, and so
        does the note below.
    sequence : {"sobol", "halton"}
        Sobol is the usual choice and better in moderate dimensions; Halton is
        simpler and degrades faster past about ten.
    scramble : bool
        Owen scrambling. Keep it on: an unscrambled set is deterministic, so it
        gives *one* answer with no way to estimate its error, and its first
        point is the corner of the cube. Scrambling preserves the low
        discrepancy and restores the ability to replicate.
    random_state : None, int or Generator
        Seeds the scrambling.

    Returns
    -------
    ndarray, shape (size, d)

    Notes
    -----
    The gain depends on the integrand. Smooth and low-dimensional: large. Kinked
    (any option payoff), high-dimensional, or discontinuous (a default
    indicator): smaller, sometimes none. Measure it with :func:`variance_ratio`
    rather than assuming.

    Examples
    --------
    The margins are uniform and the dependence is right, as for ``rvs``:

    >>> import numpy as np, rcopula as rc
    >>> from rcopula.sampling import quasi_rvs
    >>> u = quasi_rvs(rc.GumbelCopula(2.0, dim=3), 4096, random_state=0)
    >>> bool(np.all(np.abs(u.mean(axis=0) - 0.5) < 0.01))
    True
    >>> bool(abs(rc.cor_kendall(u)[0, 1] - 0.5) < 0.02)
    True

    Coverage is visibly more even -- the largest gap between consecutive sorted
    values in a margin is smaller than for independent draws:

    >>> plain = rc.GumbelCopula(2.0, dim=3).rvs(4096, random_state=0)
    >>> gap = lambda a: np.max(np.diff(np.sort(a[:, 0])))
    >>> bool(gap(u) < gap(plain))
    True
    """
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    points = _point_set(sequence, copula.dim, int(size), random_state, scramble)
    return inverse_rosenblatt(copula, points)


def antithetic_rvs(copula: Copula, size: int, *, random_state: Any = None) -> NDArray[np.float64]:
    r"""Draw from a copula in antithetic pairs.

    Each independent uniform vector :math:`z` is used twice, as :math:`z` and
    :math:`1-z`, before the inverse Rosenblatt transform. The two resulting
    draws are negatively associated, so a Monte Carlo average over them has
    lower variance **when the integrand is monotone in the uniforms**.

    Parameters
    ----------
    copula : Copula
    size : int
        Total number of draws. Rounded up to an even number, since they come in
        pairs; the first half and second half are the partners of each other, so
        ``u[i]`` pairs with ``u[size // 2 + i]``.
    random_state : None, int or Generator

    Returns
    -------
    ndarray, shape (size, d)

    Warnings
    --------
    For a payoff symmetric about the middle of the distribution -- a straddle, an
    absolute deviation, a variance -- antithetic pairing can *raise* variance
    rather than lower it, because the two halves of a pair move the same way.
    Check with :func:`variance_ratio` before adopting it.

    Examples
    --------
    >>> import numpy as np, rcopula as rc
    >>> from rcopula.sampling import antithetic_rvs
    >>> u = antithetic_rvs(rc.ClaytonCopula(2.0), 2000, random_state=0)
    >>> u.shape
    (2000, 2)

    The pairing is exact on the first coordinate, which the transform leaves
    alone:

    >>> half = 1000
    >>> bool(np.allclose(u[:half, 0] + u[half:, 0], 1.0))
    True
    """
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    half = (int(size) + 1) // 2
    rng = np.random.default_rng(random_state)
    base = rng.uniform(size=(half, copula.dim))
    doubled = np.vstack([base, 1.0 - base])
    return inverse_rosenblatt(copula, doubled)[: int(size) if size % 2 == 0 else 2 * half]


def latin_hypercube_rvs(
    copula: Copula, size: int, *, random_state: Any = None
) -> NDArray[np.float64]:
    r"""Draw from a copula using a Latin hypercube design.

    Each of the :math:`d` uniform coordinates is stratified into ``size`` equal
    bins with exactly one point per bin, then the columns are permuted
    independently. The marginal coverage is therefore guaranteed rather than
    random, which is worth having when the answer depends mostly on the margins.

    It says nothing about the *joint* structure -- that comes from the inverse
    Rosenblatt transform, as usual.

    Examples
    --------
    Every margin covers its range by construction, so the sample mean of a
    uniform coordinate is almost exactly one half:

    >>> import numpy as np, rcopula as rc
    >>> from rcopula.sampling import latin_hypercube_rvs
    >>> u = latin_hypercube_rvs(rc.FrankCopula(5.0, dim=3), 2000, random_state=0)
    >>> bool(abs(u[:, 0].mean() - 0.5) < 0.005)
    True
    >>> bool(abs(rc.cor_kendall(u)[0, 1] - rc.FrankCopula(5.0).tau()) < 0.03)
    True
    """
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    engine = qmc.LatinHypercube(d=copula.dim, seed=np.random.default_rng(random_state))
    points = np.clip(engine.random(int(size)), 1e-12, 1.0 - 1e-12)
    return inverse_rosenblatt(copula, points)


def variance_ratio(
    copula: Copula,
    payoff: Callable[[NDArray[np.float64]], ArrayLike],
    size: int,
    *,
    method: str = "sobol",
    replicates: int = 20,
    random_state: Any = None,
) -> dict[str, float]:
    r"""Measure what a variance-reduction method actually bought.

    Runs the estimator ``replicates`` times under plain Monte Carlo and again
    under ``method``, and compares the spread of the answers. This is the honest
    way to report a quasi-Monte Carlo gain: the theory gives a rate, not a
    number, and the number depends on the payoff.

    Parameters
    ----------
    copula : Copula
    payoff : callable
        Takes an ``(n, d)`` array of copula draws and returns a value per row.
        The quantity estimated is its mean.
    size : int
        Draws per replicate.
    method : {"sobol", "halton", "antithetic", "lhs"}
    replicates : int
        Independent repetitions. Each uses a different scramble or seed, which
        is what makes a spread measurable at all -- an unscrambled quasi-random
        set would give the same answer every time and no error estimate.
    random_state : None, int or Generator

    Returns
    -------
    dict
        ``"plain_se"``, ``"reduced_se"``, ``"ratio"`` (plain over reduced;
        above 1 means the method helped), ``"equivalent_sample_factor"`` (how
        many times more plain draws would be needed to match), and the two mean
        estimates, which should agree.

    Examples
    --------
    A smooth payoff, where quasi-random sampling does well:

    >>> import numpy as np, rcopula as rc
    >>> from rcopula.sampling import variance_ratio
    >>> smooth = lambda u: u[:, 0] * u[:, 1]
    >>> out = variance_ratio(rc.ClaytonCopula(2.0), smooth, 1024, replicates=12,
    ...                      random_state=0)
    >>> bool(out["ratio"] > 2.0)
    True
    >>> bool(abs(out["plain_mean"] - out["reduced_mean"]) < 0.01)
    True
    """
    if replicates < 2:
        raise ValueError(f"replicates must be at least 2, got {replicates}")
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2**63 - 1, size=(2, replicates))

    plain = np.array(
        [
            float(np.mean(np.asarray(payoff(copula.rvs(size, random_state=int(seed))))))
            for seed in seeds[0]
        ]
    )

    def draw(seed: int) -> NDArray[np.float64]:
        if method in ("sobol", "halton"):
            return quasi_rvs(copula, size, sequence=method, random_state=seed)  # type: ignore[arg-type]
        if method == "antithetic":
            return antithetic_rvs(copula, size, random_state=seed)
        if method == "lhs":
            return latin_hypercube_rvs(copula, size, random_state=seed)
        raise ValueError(f"method must be sobol, halton, antithetic or lhs; got {method!r}")

    reduced = np.array([float(np.mean(np.asarray(payoff(draw(int(seed)))))) for seed in seeds[1]])

    plain_se = float(np.std(plain, ddof=1))
    reduced_se = float(np.std(reduced, ddof=1))
    ratio = plain_se / reduced_se if reduced_se > 0 else np.inf
    return {
        "plain_mean": float(np.mean(plain)),
        "reduced_mean": float(np.mean(reduced)),
        "plain_se": plain_se,
        "reduced_se": reduced_se,
        "ratio": ratio,
        # Variance falls like 1/n under plain Monte Carlo, so matching a
        # standard-error ratio r needs r^2 times the draws.
        "equivalent_sample_factor": float(ratio**2),
    }
