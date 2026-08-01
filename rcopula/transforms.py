r"""Conditional copula distributions and the Rosenblatt transform.

The central object is the **h-function**, the conditional distribution of one
coordinate given the others:

.. math::  h(u_1 \mid u_2) = P(U_1 \le u_1 \mid U_2 = u_2)
           = \frac{\partial C(u_1, u_2)}{\partial u_2}.

It does a surprising amount of work. It is the sampler for every copula without
a closed-form generator (draw ``w``, invert ``h``); it is the building block of
vine copulas; it is the Rosenblatt transform that turns dependent data into
independent uniforms for goodness-of-fit; and in pairs trading it *is* the
signal -- :math:`h(u_1 \mid u_2)` near zero says asset 1 is unusually cheap
given where asset 2 sits, which is a statement no correlation can make.

R exposes this as ``cCopula`` and supports Archimedean and elliptical families
only. Analytic forms are used here for those, and numerical differentiation
elsewhere, so **every** family in the package is covered.

References
----------
Rosenblatt, M. (1952). Remarks on a multivariate transformation.
    *Annals of Mathematical Statistics* 23(3), 470-472.
Aas, K., Czado, C., Frigessi, A. and Bakken, H. (2009). Pair-copula
    constructions of multiple dependence.
    *Insurance: Mathematics and Economics* 44(2), 182-198.
    The h-function as the workhorse of vine copulas.
Joe, H. (2014). *Dependence Modeling with Copulas*. Chapman & Hall/CRC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri
from scipy.stats import t as student_t

from rcopula.core.archimedean import ArchimedeanCopula
from rcopula.core.base import Copula
from rcopula.core.elliptical import GaussianCopula, StudentCopula

__all__ = ["conditional_cdf", "conditional_ppf", "rosenblatt"]

#: Step for the numerical fallback. Copula CDFs are smooth in the conditioning
#: argument, so a modest step is both accurate and safe near the boundary.
_STEP = 1e-6


def _numeric_h(copula: Copula, u: NDArray[np.float64], given: int) -> NDArray[np.float64]:
    """dC/du_given by central differences, for families without a closed form."""
    hi, lo = u.copy(), u.copy()
    step = min(_STEP, 0.25 * float(min(u[:, given].min(), 1.0 - u[:, given].max())) or _STEP)
    step = max(step, 1e-10)
    hi[:, given] = np.minimum(u[:, given] + step, 1.0)
    lo[:, given] = np.maximum(u[:, given] - step, 0.0)
    return (copula.cdf(hi) - copula.cdf(lo)) / (hi[:, given] - lo[:, given])


def conditional_cdf(copula: Copula, u: ArrayLike, given: int = 1) -> NDArray[np.float64]:
    r"""Conditional distribution :math:`P(U_j \le u_j \mid U_{given} = u_{given})`.

    The h-function. For a bivariate copula, ``given=1`` returns
    :math:`h(u_1 \mid u_2) = \partial C/\partial u_2` and ``given=0`` returns
    the other conditional.

    Analytic for Archimedean, Gaussian and Student-t families; numerical
    differentiation elsewhere.

    Parameters
    ----------
    copula : Copula
        Bivariate copula.
    u : array_like
        ``(n, 2)`` points in the unit square.
    given : int
        Which coordinate to condition on.

    Returns
    -------
    ndarray
        Values in ``[0, 1]``. Under the true copula these are **uniform**,
        which is what makes them usable as a standardised signal.

    Examples
    --------
    Under independence, conditioning changes nothing:

    >>> import numpy as np
    >>> from rcopula import IndependenceCopula
    >>> from rcopula.transforms import conditional_cdf
    >>> u = np.array([[0.3, 0.8], [0.7, 0.2]])
    >>> conditional_cdf(IndependenceCopula(2), u)
    array([0.3, 0.7])

    Under positive dependence, a low value of the conditioning asset makes the
    other look *high* by comparison:

    >>> from rcopula import GaussianCopula
    >>> float(round(conditional_cdf(GaussianCopula(0.8), [[0.5, 0.05]])[0], 6))
    0.985851

    The output is uniform when the copula is right -- the property the
    Rosenblatt transform and every h-function-based test rely on:

    >>> from scipy import stats
    >>> from rcopula import ClaytonCopula
    >>> cop = ClaytonCopula(3.0)
    >>> h = conditional_cdf(cop, cop.rvs(20_000, random_state=0))
    >>> bool(stats.kstest(h, "uniform").pvalue > 0.01)
    True
    """
    arr = np.atleast_2d(np.asarray(u, dtype=np.float64))
    if arr.shape[1] != 2:
        raise ValueError(f"conditional_cdf is bivariate; got {arr.shape[1]} columns")
    if given not in (0, 1):
        raise ValueError(f"given must be 0 or 1, got {given}")

    arr = np.clip(arr, 1e-12, 1.0 - 1e-12)
    other = 1 - given
    x, cond = arr[:, other], arr[:, given]

    if isinstance(copula, ArchimedeanCopula):
        # h(x | c) = psi'(psi^-1(x) + psi^-1(c)) / psi'(psi^-1(c)); the ratio of
        # first derivatives, so the log form is stable.
        gen, theta = copula.generator, copula.theta
        t_x, t_c = gen.ipsi(x, theta), gen.ipsi(cond, theta)
        return np.clip(
            np.exp(gen.log_abs_dpsi(t_x + t_c, theta) - gen.log_abs_dpsi(t_c, theta)),
            0.0,
            1.0,
        )

    if isinstance(copula, StudentCopula):
        nu = copula.df
        rho = float(copula.sigma()[0, 1])
        a, b = student_t.ppf(x, nu), student_t.ppf(cond, nu)
        scale = np.sqrt((nu + b**2) * (1.0 - rho**2) / (nu + 1.0))
        return np.clip(student_t.cdf((a - rho * b) / scale, nu + 1.0), 0.0, 1.0)

    if isinstance(copula, GaussianCopula):
        rho = float(copula.sigma()[0, 1])
        a, b = ndtri(x), ndtri(cond)
        return np.clip(ndtr((a - rho * b) / np.sqrt(1.0 - rho**2)), 0.0, 1.0)

    return np.clip(_numeric_h(copula, arr, given), 0.0, 1.0)


def conditional_ppf(
    copula: Copula, w: ArrayLike, cond: ArrayLike, given: int = 1
) -> NDArray[np.float64]:
    r"""Invert the h-function: find ``x`` with :math:`h(x \mid \text{cond}) = w`.

    This is the conditional-distribution sampling method -- draw ``w`` uniform,
    invert, and the pair ``(x, cond)`` follows the copula. Solved by vectorised
    bisection, which handles every family uniformly and converges to 1e-15 in 50
    halvings.

    Examples
    --------
    Round-trips against :func:`conditional_cdf`:

    >>> import numpy as np
    >>> from rcopula import ClaytonCopula
    >>> from rcopula.transforms import conditional_cdf, conditional_ppf
    >>> cop = ClaytonCopula(2.5)
    >>> w = np.array([0.1, 0.5, 0.9])
    >>> cond = np.array([0.3, 0.6, 0.8])
    >>> x = conditional_ppf(cop, w, cond)
    >>> recovered = conditional_cdf(cop, np.column_stack([x, cond]))
    >>> bool(np.allclose(recovered, w, atol=1e-8))
    True
    """
    target = np.atleast_1d(np.asarray(w, dtype=np.float64))
    c = np.atleast_1d(np.asarray(cond, dtype=np.float64))
    target, c = np.broadcast_arrays(target, c)

    lo = np.full(target.shape, 1e-12)
    hi = np.full(target.shape, 1.0 - 1e-12)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        pair = np.column_stack([mid, c]) if given == 1 else np.column_stack([c, mid])
        below = conditional_cdf(copula, pair, given) < target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def rosenblatt(copula: Copula, u: ArrayLike) -> NDArray[np.float64]:
    r"""Rosenblatt transform (R's ``cCopula``).

    Maps dependent uniforms to **independent** uniforms by conditioning
    successively:

    .. math::
        Z_1 = U_1,\quad Z_2 = C(U_2 \mid U_1),\quad
        Z_3 = C(U_3 \mid U_1, U_2),\ \dots

    If the copula is correct the result is a sample of independent standard
    uniforms, which turns any goodness-of-fit question into a test of
    independence and uniformity -- the basis of the ``SnB`` and ``SnC``
    statistics.

    Implemented analytically for Archimedean and elliptical families in any
    dimension, and for arbitrary bivariate copulas.

    Examples
    --------
    The right copula gives independent uniforms; the wrong one does not:

    >>> import numpy as np
    >>> from scipy import stats
    >>> from rcopula import ClaytonCopula
    >>> from rcopula.transforms import rosenblatt
    >>> cop = ClaytonCopula(3.0, dim=3)
    >>> z = rosenblatt(cop, cop.rvs(5000, random_state=0))
    >>> bool(all(stats.kstest(z[:, j], "uniform").pvalue > 0.01 for j in range(3)))
    True
    >>> bool(abs(np.corrcoef(z[:, 0], z[:, 1])[0, 1]) < 0.05)
    True

    Transformed with the wrong parameter, the output is visibly non-uniform:

    >>> bad = rosenblatt(ClaytonCopula(0.2, dim=3), cop.rvs(5000, random_state=0))
    >>> bool(stats.kstest(bad[:, 1], "uniform").pvalue < 1e-6)
    True
    """
    arr = np.clip(np.atleast_2d(np.asarray(u, dtype=np.float64)), 1e-12, 1.0 - 1e-12)
    d = arr.shape[1]
    if d != copula.dim:
        raise ValueError(f"u has {d} columns but the copula has dim={copula.dim}")

    out = np.empty_like(arr)
    out[:, 0] = arr[:, 0]

    if isinstance(copula, ArchimedeanCopula):
        # C(u_k | u_1..u_{k-1}) = psi^(k-1)(S_k) / psi^(k-1)(S_{k-1}), with
        # S_k the running sum of inverse-generator values.
        gen, theta = copula.generator, copula.theta
        t = gen.ipsi(arr, theta)
        running = np.cumsum(t, axis=1)
        for k in range(1, d):
            out[:, k] = np.clip(
                np.exp(
                    gen.log_abs_dpsi_d(running[:, k], theta, k)
                    - gen.log_abs_dpsi_d(running[:, k - 1], theta, k)
                ),
                0.0,
                1.0,
            )
        return out

    if isinstance(copula, GaussianCopula | StudentCopula):
        is_t = isinstance(copula, StudentCopula)
        nu = copula.df if isinstance(copula, StudentCopula) else 0.0
        sigma = copula.sigma()
        x = student_t.ppf(arr, nu) if is_t else ndtri(arr)

        for k in range(1, d):
            # Condition coordinate k on the preceding ones through the usual
            # Gaussian/t partial-regression formulae.
            s11 = sigma[:k, :k]
            s12 = sigma[:k, k]
            solved = np.linalg.solve(s11, s12)
            mean = x[:, :k] @ solved
            var = float(sigma[k, k] - s12 @ solved)
            if is_t:
                quad = np.einsum("ij,ij->i", x[:, :k], np.linalg.solve(s11, x[:, :k].T).T)
                scale = np.sqrt(var * (nu + quad) / (nu + k))
                out[:, k] = student_t.cdf((x[:, k] - mean) / scale, nu + k)
            else:
                out[:, k] = ndtr((x[:, k] - mean) / np.sqrt(var))
        return np.clip(out, 0.0, 1.0)

    if d == 2:
        out[:, 1] = conditional_cdf(copula, arr, given=0)
        return out

    raise NotImplementedError(
        f"the Rosenblatt transform for {copula.name} copulas is implemented for "
        "dim=2 only; Archimedean and elliptical families support any dimension"
    )
