r"""Specification tests for copulas.

These answer questions *about the dependence structure* that come before
choosing a family, and each one narrows the search:

``indep_test``
    Is there any dependence at all? If not, nothing else matters.
``exch_test``
    Is the copula **exchangeable**, :math:`C(u,v) = C(v,u)`? Every Archimedean
    and elliptical family is. Rejecting exchangeability rules all of them out
    at once and points at the Khoudraji device or a rotated family.
``rad_sym_test``
    Is the copula **radially symmetric**, equal to its own survival copula?
    Gaussian, Student-t and Frank are; Clayton, Gumbel and Joe are not.
    Rejecting radial symmetry means the tails differ, which is exactly what a
    risk model needs to get right.
``ev_test``
    Is the copula an **extreme-value** copula? The natural null when modelling
    componentwise maxima.

Null distributions are obtained by **randomisation** rather than by R's
multiplier bootstrap. Each null hypothesis here states an invariance --
exchangeability says :math:`(U,V)` and :math:`(V,U)` have the same law, radial
symmetry says :math:`(U,V)` and :math:`(1-U,1-V)` do -- so applying that
transformation at random to each observation generates exact draws from the
null. This is simpler than the multiplier process, needs no derivative
estimates, and is valid in finite samples rather than asymptotically. The
statistics match R's; the p-values are computed differently and agree only up to
Monte-Carlo error.

References
----------
Genest, C., Neslehova, J. and Quessy, J.-F. (2012). Tests of symmetry for
    bivariate copulas. *Annals of the Institute of Statistical Mathematics*
    64(4), 811-834.
Genest, C. and Neslehova, J. G. (2014). On tests of radial symmetry for
    bivariate copulas. *Statistical Papers* 55(4), 1107-1119.
Genest, C., Remillard, B. (2004). Test of independence and randomness based on
    the empirical copula process. *Test* 13(2), 335-370.
Kojadinovic, I. (2014). Some copula inference procedures adapted to the
    presence of ties. Overview of nonparametric tests of extreme-value
    dependence. *International Statistical Review*.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rcopula.dependence import pseudo_obs
from rcopula.gof.statistics import empirical_copula_at

__all__ = [
    "DependogramResult",
    "TestResult",
    "dependogram",
    "ev_test",
    "exch_test",
    "indep_test",
    "rad_sym_test",
    "serial_indep_test",
]


class TestResult(NamedTuple):
    """Outcome of a specification test.

    Exposes ``statistic`` and ``pvalue``, as ``scipy.stats`` tests do.
    """

    statistic: float
    pvalue: float
    null: str
    n_rep: int

    def __repr__(self) -> str:
        return (
            f"TestResult(statistic={self.statistic:.6g}, pvalue={self.pvalue:.4g}, "
            f"null={self.null!r})"
        )


# The name starts with "Test", so pytest would otherwise try to collect it as a
# test class and warn. NamedTuple has no room for the attribute in its body.
TestResult.__test__ = False  # type: ignore[attr-defined]


def _prepare(data: ArrayLike, ties_method: str) -> NDArray[np.float64]:
    u = np.asarray(pseudo_obs(np.asarray(data, dtype=np.float64), ties_method=ties_method))
    if u.shape[1] != 2:
        raise ValueError(f"this test is bivariate; got {u.shape[1]} columns")
    return u


def _pesarin(replicates: NDArray[np.float64], observed: float) -> float:
    return float((0.5 + np.sum(replicates >= observed)) / (replicates.size + 1))


def _rng(random_state: np.random.Generator | int | None) -> np.random.Generator:
    return (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )


# ======================================================================
# Exchangeability
# ======================================================================


def _exch_statistic(u: NDArray[np.float64]) -> float:
    swapped = u[:, ::-1]
    return float(np.sum((empirical_copula_at(u) - empirical_copula_at(u, swapped)) ** 2))


def exch_test(
    data: ArrayLike,
    n_rep: int = 1000,
    random_state: np.random.Generator | int | None = None,
    ties_method: str = "average",
) -> TestResult:
    r"""Test whether the copula is exchangeable (R's ``exchTest``).

    :math:`H_0: C(u,v) = C(v,u)`, tested with
    :math:`S_n = \sum_i \{C_n(U_i, V_i) - C_n(V_i, U_i)\}^2`.

    Rejecting is informative out of proportion to its cost: **every Archimedean
    and every elliptical copula is exchangeable**, so a rejection eliminates all
    of them and directs you to asymmetric constructions such as the Khoudraji
    device.

    Examples
    --------
    An exchangeable copula is not rejected:

    >>> from rcopula import ClaytonCopula, exch_test
    >>> x = ClaytonCopula(3.0).rvs(300, random_state=0)
    >>> bool(exch_test(x, n_rep=200, random_state=1).pvalue > 0.05)
    True

    A deliberately asymmetric construction is:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> u = ClaytonCopula(6.0).rvs(400, random_state=2)
    >>> u[:, 1] = np.where(u[:, 0] > 0.5, rng.uniform(size=400), u[:, 1])
    >>> bool(exch_test(u, n_rep=200, random_state=1).pvalue < 0.05)
    True
    """
    u = _prepare(data, ties_method)
    rng = _rng(random_state)
    observed = _exch_statistic(u)

    # Under exchangeability (U, V) and (V, U) are equally likely, so swapping a
    # random subset of the observations produces an exact draw from the null.
    replicates = np.empty(n_rep)
    for b in range(n_rep):
        swap = np.asarray(rng.random(u.shape[0])) < 0.5
        shuffled = np.where(swap[:, None], u[:, ::-1], u)
        replicates[b] = _exch_statistic(np.asarray(pseudo_obs(shuffled)))

    return TestResult(observed, _pesarin(replicates, observed), "exchangeability", n_rep)


# ======================================================================
# Radial symmetry
# ======================================================================


def _rad_sym_statistic(u: NDArray[np.float64]) -> float:
    r"""Compare :math:`C_n` with the empirical copula of the *reflected* sample.

    The analytic survival transform
    :math:`\hat C_n(u,v) = u + v - 1 + C_n(1-u, 1-v)` looks equivalent but is
    not: :math:`C_n(1-u, 1-v)` counts observations with :math:`U_i \le 1-u`,
    which is the wrong side of the reflection. The two differ by boundary terms
    of order :math:`1/n` -- enough to shift the statistic by 10-40% at
    :math:`n = 400`. Building the empirical copula of :math:`1 - U_i` directly
    avoids the issue and reproduces R exactly.
    """
    reflected = 1.0 - u
    below = np.all(reflected[None, :, :] <= u[:, None, :], axis=2)
    survival = below.sum(axis=1) / u.shape[0]
    return float(np.sum((empirical_copula_at(u) - survival) ** 2))


def rad_sym_test(
    data: ArrayLike,
    n_rep: int = 1000,
    random_state: np.random.Generator | int | None = None,
    ties_method: str = "average",
) -> TestResult:
    r"""Test whether the copula is radially symmetric (R's ``radSymTest``).

    :math:`H_0`: :math:`(U,V)` and :math:`(1-U,1-V)` have the same distribution,
    equivalently :math:`C` equals its own survival copula.

    This is the test that separates the families whose tails behave alike from
    those whose do not. Gaussian, Student-t and Frank are radially symmetric;
    Clayton, Gumbel and Joe are not. For risk work the asymmetry *is* the
    modelling question -- a copula symmetric in the tails cannot represent
    "crashes cluster but rallies do not".

    Examples
    --------
    Frank is radially symmetric:

    >>> from rcopula import FrankCopula, rad_sym_test
    >>> x = FrankCopula(5.0).rvs(400, random_state=0)
    >>> bool(rad_sym_test(x, n_rep=200, random_state=1).pvalue > 0.05)
    True

    Clayton, with its lower-tail dependence, is not:

    >>> from rcopula import ClaytonCopula
    >>> x = ClaytonCopula(6.0).rvs(400, random_state=0)
    >>> bool(rad_sym_test(x, n_rep=200, random_state=1).pvalue < 0.05)
    True
    """
    u = _prepare(data, ties_method)
    rng = _rng(random_state)
    observed = _rad_sym_statistic(u)

    # Under radial symmetry an observation and its reflection are equally
    # likely, so reflecting a random subset draws exactly from the null.
    replicates = np.empty(n_rep)
    for b in range(n_rep):
        flip = np.asarray(rng.random(u.shape[0])) < 0.5
        reflected = np.where(flip[:, None], 1.0 - u, u)
        replicates[b] = _rad_sym_statistic(np.asarray(pseudo_obs(reflected)))

    return TestResult(observed, _pesarin(replicates, observed), "radial symmetry", n_rep)


# ======================================================================
# Independence
# ======================================================================


def _indep_statistic(u: NDArray[np.float64]) -> float:
    r"""Cramer-von Mises distance from the independence copula.

    :math:`S_n = \sum_i \{C_n(U_i) - \prod_j U_{ij}\}^2`.
    """
    return float(np.sum((empirical_copula_at(u) - np.prod(u, axis=1)) ** 2))


def indep_test(
    data: ArrayLike,
    n_rep: int = 1000,
    random_state: np.random.Generator | int | None = None,
    ties_method: str = "average",
) -> TestResult:
    r"""Test for independence (R's ``indepTest``).

    :math:`H_0: C = \Pi`, tested with a Cramer-von Mises distance between the
    empirical copula and the independence copula.

    The null is generated by **permuting each column independently**, which
    destroys any dependence while leaving the margins untouched -- an exact
    randomisation test rather than an asymptotic approximation. R's
    ``indepTestSim`` instead simulates the limiting distribution, which costs
    :math:`O(N n^2 p)` time and :math:`O(n^2 p)` memory and becomes infeasible
    for large ``n``; permutation has neither problem.

    Works in any dimension, unlike the other tests in this module.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula import ClaytonCopula, IndependenceCopula, indep_test
    >>> rng = np.random.default_rng(0)
    >>> indep = IndependenceCopula(3).rvs(400, random_state=0)
    >>> bool(indep_test(indep, n_rep=200, random_state=1).pvalue > 0.05)
    True
    >>> dep = ClaytonCopula(1.0, dim=3).rvs(400, random_state=0)
    >>> bool(indep_test(dep, n_rep=200, random_state=1).pvalue < 0.05)
    True

    It detects dependence that Pearson correlation misses entirely -- the
    textbook demonstration that "uncorrelated" and "independent" are different
    claims:

    >>> x = np.random.default_rng(7).normal(size=2000)
    >>> quadratic = np.column_stack([x, x**2])
    >>> bool(abs(np.corrcoef(x, x**2)[0, 1]) < 0.05)
    True
    >>> bool(indep_test(quadratic, n_rep=200, random_state=1).pvalue < 0.01)
    True
    """
    u = np.asarray(pseudo_obs(np.asarray(data, dtype=np.float64), ties_method=ties_method))
    rng = _rng(random_state)
    observed = _indep_statistic(u)

    replicates = np.empty(n_rep)
    for b in range(n_rep):
        shuffled = np.column_stack([rng.permutation(u[:, j]) for j in range(u.shape[1])])
        replicates[b] = _indep_statistic(shuffled)

    return TestResult(observed, _pesarin(replicates, observed), "independence", n_rep)


# ======================================================================
# Extreme-value dependence
# ======================================================================


def _ev_statistic(u: NDArray[np.float64]) -> float:
    r"""Distance between :math:`C_n` and the extreme-value copula it implies.

    An extreme-value copula satisfies the max-stability relation
    :math:`C(u^{1/m}, v^{1/m})^m = C(u, v)` for every :math:`m > 0`. Departure
    is measured at :math:`m = 2`, where the discrepancy is largest for the
    families that fail it.
    """
    root = np.sqrt(u)
    return float(np.sum((empirical_copula_at(u, root) ** 2 - empirical_copula_at(u)) ** 2))


def ev_test(
    data: ArrayLike,
    n_rep: int = 1000,
    random_state: np.random.Generator | int | None = None,
    ties_method: str = "average",
) -> TestResult:
    r"""Test whether the copula is an extreme-value copula (R's ``evTestC``).

    :math:`H_0`: :math:`C` is max-stable, :math:`C(u^{1/m}, v^{1/m})^m = C(u,v)`.

    The null distribution comes from a **parametric bootstrap through the
    Gumbel family**: Gumbel is extreme-value, so resampling from a Gumbel fitted
    to the data's Kendall tau gives draws satisfying the null with comparable
    dependence strength. This is cruder than R's multiplier construction and is
    best read as indicative.

    Examples
    --------
    Gumbel is an extreme-value copula; Clayton is not:

    >>> from rcopula import ClaytonCopula, GumbelCopula, ev_test
    >>> x = GumbelCopula(3.0).rvs(400, random_state=0)
    >>> bool(ev_test(x, n_rep=200, random_state=1).pvalue > 0.05)
    True
    >>> x = ClaytonCopula(5.0).rvs(400, random_state=0)
    >>> bool(ev_test(x, n_rep=200, random_state=1).pvalue < 0.05)
    True
    """
    from rcopula.core.archimedean import GumbelCopula

    u = _prepare(data, ties_method)
    rng = _rng(random_state)
    observed = _ev_statistic(u)

    from scipy import stats

    tau = float(np.clip(stats.kendalltau(u[:, 0], u[:, 1]).statistic, 1e-6, 0.99))
    null_copula = GumbelCopula.from_tau(tau)

    replicates = np.empty(n_rep)
    for b in range(n_rep):
        sample = np.asarray(pseudo_obs(null_copula.rvs(u.shape[0], random_state=rng)))
        replicates[b] = _ev_statistic(sample)

    return TestResult(observed, _pesarin(replicates, observed), "extreme-value dependence", n_rep)


# ======================================================================
# Independence, subset by subset
# ======================================================================


@dataclass(frozen=True)
class DependogramResult:
    """Independence tested on every subset of the coordinates.

    Attributes
    ----------
    subsets : list of tuple
        Every subset of size two or more, in the order the statistics are
        given -- :math:`2^d - d - 1` of them.
    statistics : ndarray
        The Cramer-von Mises statistic for each subset.
    pvalues : ndarray
        Permutation p-values, one per subset.
    global_pvalue : float
        A single test combining all subsets, by the minimum p-value with a
        Bonferroni-style correction through the permutation distribution -- so
        it is exact rather than conservative.
    n_rep : int
    """

    subsets: list[tuple[int, ...]]
    statistics: NDArray[np.float64]
    pvalues: NDArray[np.float64]
    global_pvalue: float
    n_rep: int

    def significant(self, level: float = 0.05) -> list[tuple[int, ...]]:
        """Subsets whose p-value falls below ``level``."""
        return [s for s, p in zip(self.subsets, self.pvalues, strict=True) if p < level]

    def summary(self, level: float = 0.05) -> str:
        """A printable report, most significant subset first."""
        order = np.argsort(self.pvalues)
        lines = [
            f"Dependogram, {self.n_rep} permutations",
            "=" * 68,
            f"  {'subset':<20}{'statistic':>14}{'p-value':>12}",
        ]
        for k in order:
            mark = " *" if self.pvalues[k] < level else ""
            label = "{" + ",".join(str(j) for j in self.subsets[k]) + "}"
            lines.append(f"  {label:<20}{self.statistics[k]:>14.6f}{self.pvalues[k]:>12.4f}{mark}")
        lines += ["", f"  global p-value       {self.global_pvalue:.4f}"]
        return "\n".join(lines)


def _mobius_matrices(u: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""``D[j, k, i] = 1{u_kj <= u_ij} - u_ij``, one matrix per coordinate.

    The Mobius transform of the empirical copula over a subset :math:`A`,

    .. math::
        M_A(\mathbf u) = \sum_{B \subseteq A} (-1)^{|A \setminus B|}
                         C_n(\mathbf u_B) \prod_{j \in A \setminus B} u_j,

    factorises: the alternating sum over subsets is exactly the expansion of a
    product, so evaluated at the data points it collapses to

    .. math::
        M_A(\mathbf U_i) = \frac1n \sum_k \prod_{j \in A}
            \bigl(\mathbf 1\{U_{kj} \le U_{ij}\} - U_{ij}\bigr).

    That turns :math:`2^{|A|}` empirical-copula evaluations into one product of
    precomputed matrices, which is what makes testing all :math:`2^d - d - 1`
    subsets affordable.
    """
    return np.stack(
        [
            (u[:, j][:, None] <= u[:, j][None, :]).astype(float) - u[:, j][None, :]
            for j in range(u.shape[1])
        ]
    )


def _subset_statistics(
    matrices: NDArray[np.float64], subsets: list[tuple[int, ...]]
) -> NDArray[np.float64]:
    out = np.empty(len(subsets))
    for position, subset in enumerate(subsets):
        product = matrices[subset[0]].copy()
        for j in subset[1:]:
            product *= matrices[j]
        out[position] = float(np.mean(np.mean(product, axis=0) ** 2))
    return out


def dependogram(
    data: ArrayLike,
    n_rep: int = 500,
    random_state: np.random.Generator | int | None = None,
    ties_method: str = "average",
) -> DependogramResult:
    r"""Test independence on every subset of the coordinates (R's ``dependogram``).

    :func:`indep_test` answers "are these independent?" with one number. This
    answers "and *which* of them are not?", by decomposing the empirical copula
    process into one component per subset (Genest and Remillard 2004). The
    components are asymptotically independent under the null, which is what
    makes reading them separately legitimate.

    It routinely finds structure a global test cannot report: three variables
    can be pairwise independent and jointly dependent, and only the
    three-element subset shows it.

    Parameters
    ----------
    data : array_like, shape (n, d)
    n_rep : int
        Permutations. The null is generated by permuting each column
        independently, which is exact rather than asymptotic.
    random_state : None, int or Generator
    ties_method : str
        Passed to :func:`~rcopula.pseudo_obs`.

    Returns
    -------
    DependogramResult

    Notes
    -----
    Cost is :math:`O(n^2 2^d)` per permutation, because each subset needs an
    :math:`n \times n` matrix product. That is fine to a few hundred
    observations in a handful of dimensions and prohibitive beyond; subsample
    rather than wait.

    **Ties cost power.** The statistic is built from the empirical copula, and
    heavily tied margins make that coarse. The same pairwise-independent
    construction below is detected easily with continuous margins and not at all
    when the coordinates take two values each -- so for count or binary data,
    read a non-significant subset as uninformative rather than as evidence of
    independence.

    Examples
    --------
    Pairwise independent and jointly dependent -- the case a global test can
    detect but not locate. With :math:`Z = (X + Y) \bmod 1` every pair is
    exactly independent, while the triple is deterministic:

    >>> import numpy as np
    >>> from rcopula.htest import dependogram
    >>> rng = np.random.default_rng(0)
    >>> x, y = rng.uniform(size=(2, 400))
    >>> data = np.column_stack([x, y, (x + y) % 1.0])
    >>> result = dependogram(data, n_rep=200, random_state=1)
    >>> result.significant()
    [(0, 1, 2)]

    Genuinely independent data leaves every subset alone:

    >>> quiet = dependogram(rng.uniform(size=(300, 3)), n_rep=200, random_state=1)
    >>> quiet.significant()
    []
    """
    u = np.asarray(pseudo_obs(np.asarray(data, dtype=np.float64), ties_method=ties_method))
    n, d = u.shape
    if d < 2:
        raise ValueError(f"a dependogram needs at least 2 columns, got {d}")
    rng = _rng(random_state)

    subsets = [tuple(s) for size in range(2, d + 1) for s in itertools.combinations(range(d), size)]
    matrices = _mobius_matrices(u)
    observed = _subset_statistics(matrices, subsets)

    replicates = np.empty((n_rep, len(subsets)))
    for b in range(n_rep):
        # Permuting a column permutes its matrix by the same permutation in
        # both axes, so the matrices are reindexed rather than rebuilt.
        permuted = np.empty_like(matrices)
        for j in range(d):
            order = rng.permutation(n)
            permuted[j] = matrices[j][np.ix_(order, order)]
        replicates[b] = _subset_statistics(permuted, subsets)

    pvalues = np.array([_pesarin(replicates[:, k], observed[k]) for k in range(len(subsets))])
    # The global test uses the smallest p-value, calibrated against the smallest
    # p-value each permutation would have produced -- which accounts for the
    # multiplicity exactly instead of assuming independence between subsets.
    replicate_min = np.array(
        [
            min(_pesarin(replicates[:, k], replicates[b, k]) for k in range(len(subsets)))
            for b in range(n_rep)
        ]
    )
    global_pvalue = float((0.5 + np.sum(replicate_min <= pvalues.min())) / (n_rep + 1))

    return DependogramResult(
        subsets=subsets,
        statistics=observed,
        pvalues=pvalues,
        global_pvalue=global_pvalue,
        n_rep=n_rep,
    )


def serial_indep_test(
    x: ArrayLike,
    lags: int = 3,
    n_rep: int = 500,
    random_state: np.random.Generator | int | None = None,
    ties_method: str = "average",
) -> DependogramResult:
    r"""Test a series for serial independence (R's ``serialIndepTest``).

    Embeds the series into its own lags and runs :func:`dependogram` on the
    result, so the answer names *which lag structure* carries the dependence
    rather than only whether some does.

    The subsets are read as lag sets: ``(0, 1)`` is the dependence between
    consecutive observations, ``(0, 2)`` between observations two apart, and
    ``(0, 1, 2)`` a three-way structure that neither pair reveals -- which is
    exactly what a correlogram cannot show.

    Being rank-based, it is invariant to any increasing transform of the series
    and needs no moments, so it applies to heavy-tailed returns where the
    autocorrelation function is not even well defined.

    Parameters
    ----------
    x : array_like, shape (n,)
        A univariate series, in time order.
    lags : int
        Embedding dimension. ``3`` looks at lags 1 and 2.
    n_rep : int
        Permutations for the null.
    random_state : None, int or Generator
    ties_method : str

    Returns
    -------
    DependogramResult
        Subsets are indices into the embedded vector, so ``0`` is the current
        observation and ``k`` the one ``k`` steps back.

    Notes
    -----
    Permuting the embedded matrix column by column destroys serial dependence
    while preserving each column's marginal distribution, which is the exact
    randomisation this needs. It does **not** preserve the overlap between rows
    that embedding creates, so the test is mildly conservative -- the rows are
    not independent, and the permutation null treats them as if they were.

    Examples
    --------
    White noise passes; an AR(1) does not:

    >>> import numpy as np
    >>> from rcopula.htest import serial_indep_test
    >>> rng = np.random.default_rng(0)
    >>> noise = rng.standard_normal(600)
    >>> bool(serial_indep_test(noise, n_rep=200, random_state=1).global_pvalue > 0.05)
    True
    >>> series = np.zeros(600)
    >>> for t in range(1, 600):
    ...     series[t] = 0.6 * series[t - 1] + rng.standard_normal()
    >>> bool(serial_indep_test(series, n_rep=200, random_state=1).global_pvalue < 0.05)
    True

    A GARCH-like series is uncorrelated in level and dependent in magnitude,
    which is the case an autocorrelation function reports as clean:

    >>> volatility = np.exp(0.5 * rng.standard_normal(2000).cumsum() / 30)
    >>> returns = volatility * rng.standard_normal(2000)
    >>> bool(abs(np.corrcoef(returns[:-1], returns[1:])[0, 1]) < 0.06)
    True
    >>> bool(serial_indep_test(np.abs(returns), n_rep=200, random_state=1).global_pvalue < 0.05)
    True
    """
    series = np.asarray(x, dtype=np.float64).ravel()
    if lags < 2:
        raise ValueError(f"lags must be at least 2 to have a pair to test, got {lags}")
    if series.size <= lags:
        raise ValueError(f"need more than {lags} observations, got {series.size}")

    # Row t of the embedding is (x_t, x_{t-1}, ..., x_{t-lags+1}).
    rows = series.size - lags + 1
    embedded = np.column_stack([series[lags - 1 - k : lags - 1 - k + rows] for k in range(lags)])
    return dependogram(embedded, n_rep=n_rep, random_state=random_state, ties_method=ties_method)
