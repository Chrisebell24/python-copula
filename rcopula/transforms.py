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

__all__ = [
    "conditional_cdf",
    "conditional_ppf",
    "htrafo",
    "inverse_rosenblatt",
    "radial_cdf",
    "radial_ppf",
    "radial_simplex",
    "rosenblatt",
]

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


#: Bisection steps for the generic inverse Rosenblatt transform. Each halves
#: the bracket, so 60 takes a unit interval below 1e-18 -- past double
#: precision, and therefore as exact as the forward transform it inverts.
_BISECTION_STEPS = 60


def inverse_rosenblatt(copula: Copula, z: ArrayLike) -> NDArray[np.float64]:
    r"""Inverse Rosenblatt transform: independent uniforms to copula draws.

    The other direction of :func:`rosenblatt`, and the reason it matters is
    sampling. ``rvs`` draws its own randomness; this takes randomness you supply,
    which is what lets a **quasi-random** point set be pushed through a copula --
    see :mod:`rcopula.sampling`. It is also how conditional simulation works:
    fix the first coordinates, vary the rest.

    Parameters
    ----------
    copula : Copula
    z : array_like, shape (n, d)
        Independent uniforms.

    Returns
    -------
    ndarray, shape (n, d)
        Draws from ``copula``.

    Notes
    -----
    Elliptical copulas invert analytically. Everything else is inverted by
    bisection **against :func:`rosenblatt` itself**, so the two are exact
    inverses by construction rather than by two derivations agreeing -- which is
    the failure mode this arrangement removes. The conditional CDF is monotone
    in its own argument, so bisection cannot fail; 60 steps takes the bracket
    below double precision.

    Examples
    --------
    Round trip, in both directions:

    >>> import numpy as np, rcopula as rc
    >>> from rcopula.transforms import inverse_rosenblatt, rosenblatt
    >>> cop = rc.ClaytonCopula(3.0, dim=3)
    >>> u = cop.rvs(500, random_state=0)
    >>> bool(np.max(np.abs(inverse_rosenblatt(cop, rosenblatt(cop, u)) - u)) < 1e-9)
    True

    Independent uniforms in, correctly dependent draws out:

    >>> rng = np.random.default_rng(0)
    >>> z = rng.uniform(size=(20000, 2))
    >>> u = inverse_rosenblatt(rc.GumbelCopula(2.0), z)
    >>> bool(abs(rc.cor_kendall(u)[0, 1] - 0.5) < 0.02)
    True
    """
    arr = np.clip(np.atleast_2d(np.asarray(z, dtype=np.float64)), 1e-12, 1.0 - 1e-12)
    d = arr.shape[1]
    if d != copula.dim:
        raise ValueError(f"z has {d} columns but the copula has dim={copula.dim}")

    if isinstance(copula, GaussianCopula | StudentCopula):
        is_t = isinstance(copula, StudentCopula)
        nu = copula.df if isinstance(copula, StudentCopula) else 0.0
        sigma = copula.sigma()
        x = np.empty_like(arr)
        x[:, 0] = student_t.ppf(arr[:, 0], nu) if is_t else ndtri(arr[:, 0])
        for k in range(1, d):
            s11 = sigma[:k, :k]
            s12 = sigma[:k, k]
            solved = np.linalg.solve(s11, s12)
            mean = x[:, :k] @ solved
            var = float(sigma[k, k] - s12 @ solved)
            if is_t:
                quad = np.einsum("ij,ij->i", x[:, :k], np.linalg.solve(s11, x[:, :k].T).T)
                scale = np.sqrt(var * (nu + quad) / (nu + k))
                x[:, k] = mean + scale * student_t.ppf(arr[:, k], nu + k)
            else:
                x[:, k] = mean + np.sqrt(var) * ndtri(arr[:, k])
        return np.clip(student_t.cdf(x, nu) if is_t else ndtr(x), 1e-12, 1.0 - 1e-12)

    # Filled with a valid interior point rather than np.empty: the k-th output
    # of rosenblatt depends only on coordinates 0..k, so the columns past k
    # cannot change the answer -- but they are still evaluated, and uninitialised
    # memory there produces warnings and, at 0 or 1, non-finite generator values.
    out = np.full_like(arr, 0.5)
    out[:, 0] = arr[:, 0]
    for k in range(1, d):
        low = np.full(arr.shape[0], 1e-12)
        high = np.full(arr.shape[0], 1.0 - 1e-12)
        probe = out.copy()
        for _ in range(_BISECTION_STEPS):
            middle = 0.5 * (low + high)
            probe[:, k] = middle
            value = rosenblatt(copula, probe)[:, k]
            below = value < arr[:, k]
            low = np.where(below, middle, low)
            high = np.where(below, high, middle)
        out[:, k] = 0.5 * (low + high)
    return out


# --------------------------------------------------------------------------
# the Archimedean simplex decomposition
# --------------------------------------------------------------------------
#
# References for this section:
#
# McNeil, A. J. and Neslehova, J. (2009). Multivariate Archimedean copulas,
#     d-monotone functions and l1-norm symmetric distributions.
#     *Annals of Statistics* 37(5B), 3059-3097.
#     The decomposition itself: an Archimedean copula sample is a radial
#     variable times a point drawn uniformly from the unit simplex.
# Hering, C. and Hofert, M. (2015). Goodness-of-fit tests for Archimedean
#     copulas in high dimensions. In *Innovations in Quantitative Risk
#     Management*, 357-373. Springer.
#     The transformation to independent uniforms, ``htrafo`` in R.


def radial_simplex(copula: ArchimedeanCopula, u: ArrayLike) -> tuple[NDArray, NDArray]:
    r"""Split an Archimedean sample into its radial and angular parts.

    McNeil and Neslehova showed that :math:`U \sim C` for an Archimedean copula
    with generator :math:`\psi` if and only if

    .. math::  \bigl(\psi^{-1}(U_1), \dots, \psi^{-1}(U_d)\bigr) = R\,S,

    with :math:`S` **uniform on the unit simplex** and independent of the radial
    variable :math:`R = \sum_k \psi^{-1}(U_k)`. All the family-specific
    information sits in the distribution of :math:`R`; the angular part is the
    same for every Archimedean copula in every dimension.

    That is the whole basis for testing *Archimedeanity* rather than testing one
    particular family: if the angular part is not uniform on the simplex, no
    Archimedean copula fits, whatever generator is tried.

    Parameters
    ----------
    copula : ArchimedeanCopula
        Supplies the generator. Its parameter must be set.
    u : array_like, shape (n, d)

    Returns
    -------
    radial : ndarray, shape (n,)
        :math:`R`.
    angular : ndarray, shape (n, d)
        :math:`S`, whose rows sum to one.

    Examples
    --------
    The angular part is uniform on the simplex, so each of its coordinates has
    mean :math:`1/d` -- and this does not depend on the family or the parameter:

    >>> import numpy as np, rcopula as rc
    >>> from rcopula.transforms import radial_simplex
    >>> for copula in (rc.ClaytonCopula(3.0, dim=4), rc.GumbelCopula(2.0, dim=4)):
    ...     u = copula.rvs(20000, random_state=0)
    ...     angular = radial_simplex(copula, u)[1]
    ...     print(bool(np.all(np.abs(angular.mean(axis=0) - 0.25) < 0.01)))
    True
    True
    """
    if not isinstance(copula, ArchimedeanCopula):
        raise TypeError(
            f"the radial-simplex decomposition is Archimedean; got "
            f"{type(copula).__name__}. Elliptical copulas have their own "
            "(radius, direction) decomposition on the sphere rather than the "
            "simplex."
        )
    u = np.atleast_2d(np.asarray(u, dtype=float))
    if u.shape[1] != copula.dim:
        raise ValueError(f"u has {u.shape[1]} columns but the copula has dim {copula.dim}")
    theta = float(copula.params[0])
    inverse = np.column_stack([copula.generator.ipsi(u[:, j], theta) for j in range(copula.dim)])
    radial = np.asarray(inverse.sum(axis=1), dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        angular = inverse / radial[:, None]
    return radial, np.asarray(angular, dtype=float)


def htrafo(copula: ArchimedeanCopula, u: ArrayLike) -> NDArray[np.float64]:
    r"""Hering-Hofert transform: Archimedean data to independent uniforms.

    The Rosenblatt transform (:func:`rosenblatt`) does the same job by
    conditioning one coordinate at a time, which needs :math:`d-1` derivatives
    of the generator and loses accuracy fast as :math:`d` grows. This transform
    goes through the simplex decomposition instead, so it needs no high-order
    derivatives at all and stays usable at :math:`d = 100` -- which is why it
    exists.

    Writing :math:`S_j = \sum_{k \le j} \psi^{-1}(U_k)`,

    .. math::

        Y_j = \left(\frac{S_j}{S_{j+1}}\right)^{j}, \quad j = 1, \dots, d-1,
        \qquad Y_d = K\bigl(\psi(S_d)\bigr),

    with :math:`K` the Kendall distribution function. Under the null that ``u``
    came from this copula the :math:`Y_j` are independent and uniform, so any
    test of multivariate uniformity becomes a goodness-of-fit test.

    The first :math:`d-1` components come from the angular part and the last
    from the radial part; since those are independent, so are the two groups.

    Parameters
    ----------
    copula : ArchimedeanCopula
    u : array_like, shape (n, d)

    Returns
    -------
    ndarray, shape (n, d)

    See Also
    --------
    rosenblatt : the conditioning-based alternative, exact but derivative-hungry.
    radial_simplex : the decomposition this is built on.

    Examples
    --------
    Data from the copula transforms to uniforms; data from a different copula
    does not:

    >>> import numpy as np, rcopula as rc
    >>> from scipy import stats
    >>> from rcopula.transforms import htrafo
    >>> copula = rc.ClaytonCopula(2.0, dim=5)
    >>> y = htrafo(copula, copula.rvs(4000, random_state=0))
    >>> bool(stats.kstest(y.ravel(), "uniform").pvalue > 0.01)
    True
    >>> wrong = htrafo(copula, rc.GumbelCopula(3.0, dim=5).rvs(4000, random_state=0))
    >>> bool(stats.kstest(wrong.ravel(), "uniform").pvalue < 1e-6)
    True

    It still works where the Rosenblatt transform's derivatives would not:

    >>> big = rc.GumbelCopula(2.0, dim=50)
    >>> y = htrafo(big, big.rvs(500, random_state=0))
    >>> bool(np.all(np.isfinite(y)) and y.shape == (500, 50))
    True
    """
    from rcopula.kendall import kendall_cdf

    radial, _ = radial_simplex(copula, u)
    u = np.atleast_2d(np.asarray(u, dtype=float))
    dim = copula.dim
    theta = float(copula.params[0])
    inverse = np.column_stack([copula.generator.ipsi(u[:, j], theta) for j in range(dim)])

    partial = np.cumsum(inverse, axis=1)
    out = np.empty_like(partial)
    for j in range(1, dim):
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = partial[:, j - 1] / partial[:, j]
        out[:, j - 1] = np.clip(ratio, 0.0, 1.0) ** j
    # The last component carries everything the angular part discarded: the
    # radial variable, mapped through K, which is exactly the copula's Kendall
    # distribution function evaluated at C(u).
    out[:, dim - 1] = kendall_cdf(copula, copula.generator.psi(radial, theta))
    return np.clip(out, 0.0, 1.0)


def radial_cdf(copula: ArchimedeanCopula, x: ArrayLike) -> NDArray[np.float64]:
    r"""Distribution function of the radial part (R's ``pacR``).

    :func:`radial_simplex` splits an Archimedean sample into a radial variable
    :math:`R = \sum_j \psi^{-1}(U_j)` and a point uniform on the simplex. The
    angular half is the same for every Archimedean copula in every dimension,
    so **all** the family-specific information lives in the distribution of
    :math:`R` -- which is this.

    It needs no new machinery, because it is the Kendall distribution function
    in disguise. Since :math:`C(\mathbf U) = \psi(R)` and :math:`\psi` is
    decreasing,

    .. math::

        K(t) = P\{\psi(R) \le t\} = P\{R \ge \psi^{-1}(t)\},
        \qquad\text{so}\qquad
        F_R(x) = 1 - K(\psi(x)).

    Parameters
    ----------
    copula : ArchimedeanCopula
    x : array_like
        Radii, non-negative.

    Returns
    -------
    ndarray

    Examples
    --------
    >>> import numpy as np, rcopula as rc
    >>> from rcopula.transforms import radial_cdf
    >>> cop = rc.ClaytonCopula(2.0, dim=3)
    >>> values = radial_cdf(cop, [1.5861, 11.6107, 346.2882])
    >>> np.round(values, 3)
    array([0.1  , 0.499, 0.9  ])

    Which is what the sampled radial part actually does:

    >>> radii = np.sum(cop.generator.ipsi(cop.rvs(50000, random_state=0), 2.0), axis=1)
    >>> bool(np.all(np.abs(np.mean(radii[:, None] <= [1.5861, 11.6107], axis=0)
    ...                    - values[:2]) < 0.01))
    True
    """
    if not isinstance(copula, ArchimedeanCopula):
        raise TypeError(
            f"the radial part is Archimedean; got {type(copula).__name__}. An "
            "elliptical copula has its own radial decomposition, on the sphere "
            "rather than the simplex."
        )
    from rcopula.kendall import kendall_cdf

    radii = np.atleast_1d(np.asarray(x, dtype=np.float64))
    if np.any(radii < 0.0):
        raise ValueError("a radius cannot be negative")
    theta = float(copula.params[0])
    generator = copula.generator
    return np.asarray(
        1.0 - np.asarray(kendall_cdf(copula, generator.psi(radii, theta)), dtype=float),
        dtype=np.float64,
    )


def radial_ppf(copula: ArchimedeanCopula, q: ArrayLike) -> NDArray[np.float64]:
    r"""Quantile function of the radial part (R's ``qacR``).

    The inverse of :func:`radial_cdf`, by the same identity: since
    :math:`F_R(x) = 1 - K(\psi(x))`, the quantile is
    :math:`\psi^{-1}(K^{-1}(1 - q))`, so it reduces to the Kendall function's
    own quantile and needs no root-finding of its own.

    Parameters
    ----------
    copula : ArchimedeanCopula
    q : array_like
        Probabilities in :math:`[0, 1]`.

    Returns
    -------
    ndarray

    Examples
    --------
    >>> import numpy as np, rcopula as rc
    >>> from rcopula.transforms import radial_cdf, radial_ppf
    >>> cop = rc.GumbelCopula(2.0, dim=4)
    >>> levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    >>> bool(np.max(np.abs(radial_cdf(cop, radial_ppf(cop, levels)) - levels)) < 1e-6)
    True
    """
    if not isinstance(copula, ArchimedeanCopula):
        raise TypeError(f"the radial part is Archimedean; got {type(copula).__name__}")
    from rcopula.kendall import kendall_ppf

    levels = np.atleast_1d(np.asarray(q, dtype=np.float64))
    if np.any(levels < 0.0) or np.any(levels > 1.0):
        raise ValueError("q must lie in [0, 1]")
    theta = float(copula.params[0])
    inner = np.asarray(kendall_ppf(copula, 1.0 - levels), dtype=np.float64)
    return np.asarray(copula.generator.ipsi(inner, theta), dtype=np.float64)
