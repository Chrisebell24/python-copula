r"""The empirical copula and its smoothed variants.

Given data :math:`X_1, \dots, X_n`, the empirical copula is the rank-based
estimator

.. math::

    C_n(\mathbf{u}) = \frac{1}{n}\sum_{i=1}^{n}
        \mathbf{1}\{\hat U_{i1} \le u_1, \dots, \hat U_{id} \le u_d\},

built from pseudo-observations. It is the nonparametric baseline every
goodness-of-fit test is measured against, and it converges to the true copula
without assuming a family.

Two smoothings improve on the raw step function:

* **Beta** (Segers, Sibuya & Tsukahara 2017) replaces each indicator with a Beta
  distribution function tied to the rank. The result is a genuine copula, is
  continuous, and -- unlike the raw estimator -- has a **density**.
* **Checkerboard** spreads each observation uniformly over a grid cell. Also a
  genuine copula, and the natural choice when there are ties.

Both dominate the raw estimator in mean squared error, markedly so at small
sample sizes.

References
----------
Deheuvels, P. (1979). La fonction de dependance empirique et ses proprietes.
    *Academie Royale de Belgique, Bulletin de la Classe des Sciences* 65,
    274-292.
Segers, J., Sibuya, M. and Tsukahara, H. (2017). The empirical beta copula.
    *Journal of Multivariate Analysis* 155, 35-51.
    Equation 2.1 for the beta smoothing, 4.1 for the checkerboard.
Remillard, B. and Scaillet, O. (2009). Testing for equality between two copulas.
    *Journal of Multivariate Analysis* 100(3), 377-386.
    The finite-difference partial-derivative estimator in :meth:`dCdu`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from rcopula.core.base import Copula, TailDependence
from rcopula.dependence import beta_n, cor_kendall, cor_spearman, pseudo_obs

__all__ = ["EmpiricalCopula"]

SMOOTHINGS = ("none", "beta", "checkerboard")


class EmpiricalCopula(Copula):
    r"""A copula read directly off your data, with no assumed family.

    It describes the dependence in a sample using only the ranks of the
    observations. Use it as a model-free reference: to compare a fitted
    parametric copula against, to compute sample dependence measures, or to
    resample the dependence structure of the data.

    Parameters
    ----------
    data : array_like of float, shape (n, d)
        The observations, one row per observation and one column per variable
        (``n >= 2``, ``d >= 2``). They are converted to ranks internally
        (pseudo-observations), so raw data on any scale is fine.
    smoothing : {"none", "beta", "checkerboard"}, default "none"
        ``"none"`` is the classical step-function estimator. ``"beta"`` and
        ``"checkerboard"`` are smoothed and are genuine copulas; only ``"beta"``
        admits a density (so :meth:`pdf` works only with ``"beta"``).
    offset : float, default 0.0
        Keyword-only. Added to the denominator ``n`` of the unsmoothed
        estimator, as in R's ``empCopula``. Rarely needed.
    ties_method : str, default "average"
        Keyword-only. How tied values are ranked; passed to
        :func:`~rcopula.dependence.pseudo_obs`. Methods that keep ties
        (``"average"``, ``"min"``, ``"max"``) make the smoothed estimators
        spread each tied group over the block of ranks it occupies (see
        Notes); ``"first"`` and ``"random"`` break ties, so every observation
        keeps a rank of its own.

    Attributes
    ----------
    smoothing : str
        The smoothing in use.
    offset : float
        The denominator offset.
    ties_method : str
        The tie-breaking rule used for the ranks.
    n_obs : int
        Number of observations.
    pseudo_observations : numpy.ndarray of float, shape (n, d)
        The rank-transformed data.

    Raises
    ------
    ValueError
        If ``smoothing`` is not one of the three options, ``data`` is not a
        2-D array, there are fewer than two observations, or the data has
        fewer than two columns.
    TypeError
        If given an unknown keyword argument (they used to be silently
        ignored, so a misspelling such as ``smothing=`` went unnoticed).

    Notes
    -----
    **Ties.** The beta and checkerboard estimators are indexed by ranks
    ``1..n``. When ``k`` observations tie in a column they jointly occupy
    the ranks ``r, ..., r + k - 1``; each of them is given the *average* of
    the ``k`` beta distributions (beta smoothing) or the union of the ``k``
    grid cells (checkerboard) for those ranks, rather than one rank rounded
    from the mid-rank. This keeps the margins exactly uniform, so both
    smoothings remain genuine copulas with ties. Without ties it is the usual
    estimator.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula import ClaytonCopula, EmpiricalCopula
    >>> truth = ClaytonCopula(2.0)
    >>> x = truth.rvs(2000, random_state=0)
    >>> emp = EmpiricalCopula(x)
    >>> grid = np.array([[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]])
    >>> bool(np.max(np.abs(emp.cdf(grid) - truth.cdf(grid))) < 0.02)
    True

    The beta-smoothed version has a density, which the raw estimator does not:

    >>> smooth = EmpiricalCopula(x, smoothing="beta")
    >>> bool(np.all(smooth.pdf(grid) > 0))
    True
    >>> EmpiricalCopula(x).pdf(grid)
    Traceback (most recent call last):
        ...
    NotImplementedError: the unsmoothed empirical copula is a step function...

    Dependence measures come from the sample, not from a parametric form:

    >>> bool(abs(emp.tau() - truth.tau()) < 0.05)
    True
    """

    name = "Empirical"
    param_names: tuple[str, ...] = ()

    def __init__(
        self,
        data: ArrayLike,
        smoothing: str = "none",
        *,
        offset: float = 0.0,
        ties_method: str = "average",
    ) -> None:
        if smoothing not in SMOOTHINGS:
            raise ValueError(f"smoothing must be one of {SMOOTHINGS}, got {smoothing!r}")

        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(
                "data must be a 2-D array of shape (n_observations, n_variables) with at "
                f"least two variables; got a {arr.ndim}-D array of shape {arr.shape}"
            )
        if arr.shape[0] < 2:
            raise ValueError("an empirical copula needs at least two observations")

        self.smoothing = smoothing
        self.offset = float(offset)
        self.ties_method = ties_method
        self._u = np.asarray(pseudo_obs(arr, ties_method=ties_method), dtype=np.float64)
        self._n = arr.shape[0]
        # The smoothed estimators are indexed by ranks 1..n. Observation i in
        # column j occupies the block of ranks lo+1..hi: a single rank without
        # ties, the k ranks a tied group shares otherwise. Ranking the pseudo-
        # observations again ("min"/"max") recovers those blocks whatever
        # ties_method produced them.
        self._rank_lo = np.column_stack(
            [stats.rankdata(col, method="min") - 1 for col in self._u.T]
        ).astype(np.int64)
        self._rank_hi = np.column_stack(
            [stats.rankdata(col, method="max") for col in self._u.T]
        ).astype(np.int64)
        self._has_ties = bool(np.any(self._rank_hi - self._rank_lo > 1))

        super().__init__(np.empty(0), arr.shape[1])

    # -- properties -----------------------------------------------------

    @property
    def n_obs(self) -> int:
        """Number of observations (rows of ``data``) the estimator was built from.

        Returns
        -------
        int
        """
        return self._n

    @property
    def pseudo_observations(self) -> NDArray[np.float64]:
        """The data converted to ranks scaled into ``(0, 1)``.

        These pseudo-observations are what the estimator is actually built
        from; each column is roughly uniform.

        Returns
        -------
        numpy.ndarray of float, shape (n, d)
        """
        return self._u

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        """Parameter bounds; always empty, since the estimator has no parameters.

        Returns
        -------
        list
            ``[]``.
        """
        return []

    def _reconstruct(self, params: ArrayLike, free: ArrayLike) -> EmpiricalCopula:
        return EmpiricalCopula(
            self._u, self.smoothing, offset=self.offset, ties_method=self.ties_method
        )

    # -- numerical core -------------------------------------------------

    def _cdf(self, u, params):
        n = self._n
        if self.smoothing == "none":
            # Mean over observations of the indicator that all coordinates fall
            # below u. The comparison is (n_eval, n_obs, d), so chunk if huge.
            below = np.all(self._u[None, :, :] <= u[:, None, :], axis=2)
            return below.sum(axis=1) / (n + self.offset)

        lo, hi = self._rank_lo, self._rank_hi
        if self.smoothing == "beta":
            # C_n^beta(u) = (1/n) sum_i prod_j pbeta(u_j; R_ij, n - R_ij + 1)
            out = np.empty(u.shape[0])
            for k, point in enumerate(u):
                if self._has_ties:
                    terms = self._block_mean(point, stats.binom.sf, n)
                else:
                    terms = stats.beta.cdf(point[None, :], hi, n - hi + 1)
                out[k] = np.prod(terms, axis=1).mean()
            return out

        # Checkerboard: each observation is spread uniformly over its block of
        # grid cells -- one cell, or the k cells a tied group shares.
        out = np.empty(u.shape[0])
        for k, point in enumerate(u):
            terms = np.clip((n * point[None, :] - lo) / (hi - lo), 0.0, 1.0)
            out[k] = np.prod(terms, axis=1).mean()
        return out

    def _block_mean(
        self, point: NDArray[np.float64], kernel: Callable[..., Any], n_trials: int
    ) -> NDArray[np.float64]:
        r"""Average of a rank-indexed kernel over each observation's block of ranks.

        For the beta smoothing the kernel for rank ``m`` is
        ``pbeta(u; m, n + 1 - m) = P(Binomial(n, u) >= m)`` (``kernel =
        binom.sf`` at ``m - 1``), or its density
        ``dbeta(u; m, n + 1 - m) = n * dbinom(m - 1; n - 1, u)``. Cumulative
        sums over ``m = 1..n`` give every block's average in one pass per
        coordinate.
        """
        lo, hi = self._rank_lo, self._rank_hi
        out = np.empty(lo.shape)
        m_minus_1 = np.arange(self._n)
        for j in range(self._dim):
            values = kernel(m_minus_1, n_trials, point[j])
            cum = np.concatenate([[0.0], np.cumsum(values)])
            out[:, j] = (cum[hi[:, j]] - cum[lo[:, j]]) / (hi[:, j] - lo[:, j])
        return out

    def _logpdf(self, u, params):
        if self.smoothing != "beta":
            raise NotImplementedError(
                "the unsmoothed empirical copula is a step function and the "
                "checkerboard estimator is piecewise uniform, so neither has a "
                "density; use smoothing='beta' if you need one"
            )
        n = self._n
        hi = self._rank_hi
        out = np.empty(u.shape[0])
        for k, point in enumerate(u):
            if self._has_ties:
                terms = n * self._block_mean(point, stats.binom.pmf, n - 1)
            else:
                terms = stats.beta.pdf(point[None, :], hi, n - hi + 1)
            out[k] = np.prod(terms, axis=1).mean()
        with np.errstate(divide="ignore"):
            return np.log(out)

    def _rvs(self, size, params, rng):
        """Resample the pseudo-observations, smoothing if requested."""
        idx = rng.integers(0, self._n, size=size)
        if self.smoothing == "none":
            return self._u[idx]
        lo, hi = self._rank_lo[idx], self._rank_hi[idx]
        if self.smoothing == "beta":
            r = hi
            if self._has_ties:
                # A rank drawn uniformly from the observation's block. Only
                # done with ties, so untied data keeps its random stream.
                step = np.floor(rng.uniform(size=lo.shape) * (hi - lo)).astype(np.int64)
                r = np.minimum(lo + 1 + step, hi)
            return rng.beta(r, self._n - r + 1)
        # Checkerboard: uniform within the selected block of cells.
        return (hi - rng.uniform(size=(size, self._dim)) * (hi - lo)) / self._n

    # -- estimators R exposes as free functions -------------------------

    def dCdu(self, u: ArrayLike, bandwidth: float | None = None) -> NDArray[np.float64]:
        r"""Estimate how fast the empirical copula changes in each coordinate.

        Returns approximate partial derivatives
        :math:`\partial C_n/\partial u_j` at the given points, by finite
        differences (R's ``dCn``). Mostly needed by goodness-of-fit tests (the
        multiplier bootstrap), since a step function has no true derivatives.

        Parameters
        ----------
        u : array_like of float, shape (m, d) or (d,)
            Points in the unit cube at which to evaluate.
        bandwidth : float or None, default None
            Half-width ``b`` of the difference step, in ``(0, 1)``. ``None`` (or
            0) uses :math:`n^{-1/2}`.

        Returns
        -------
        numpy.ndarray of float, shape (m, d)
            Column ``j`` holds the estimated derivative with respect to
            ``u_j``.

        Raises
        ------
        ValueError
            If ``u`` has the wrong number of columns.

        Notes
        -----
        Uses the Remillard-Scaillet (2009) central difference

        .. math::
            \frac{C_n(\dots, u_j + b, \dots) - C_n(\dots, u_j - b, \dots)}{2b},

        with default bandwidth :math:`b = n^{-1/2}`. These are what the
        multiplier bootstrap needs, since the true derivatives of a step
        function do not exist.

        Examples
        --------
        >>> import numpy as np
        >>> from rcopula import EmpiricalCopula, IndependenceCopula
        >>> x = IndependenceCopula(2).rvs(20_000, random_state=0)
        >>> d = EmpiricalCopula(x).dCdu([[0.5, 0.5]])
        >>> bool(np.allclose(d, 0.5, atol=0.05))       # dC/du = v = 0.5
        True
        """
        arr = self._validate_u(u)
        b = float(bandwidth) if bandwidth else 1.0 / np.sqrt(self._n)
        out = np.empty_like(arr)
        for j in range(self._dim):
            hi = arr.copy()
            lo = arr.copy()
            hi[:, j] = np.minimum(arr[:, j] + b, 1.0)
            lo[:, j] = np.maximum(arr[:, j] - b, 0.0)
            out[:, j] = (self.cdf(hi) - self.cdf(lo)) / (hi[:, j] - lo[:, j])
        return out

    # -- dependence measures, taken from the sample ---------------------

    def tau(self) -> float:
        """Kendall's tau of the data: a rank correlation between -1 and 1.

        It is the probability that two observations are ordered the same way
        in both variables minus the probability they are ordered oppositely.

        Returns
        -------
        float
            The sample value; for ``dim > 2``, the average over all pairs.
        """
        m = cor_kendall(self._u)
        return float(m[np.triu_indices(self._dim, 1)].mean())

    def rho(self) -> float:
        """Spearman's rho of the data: the correlation of the ranks, between -1 and 1.

        Returns
        -------
        float
            The sample value; for ``dim > 2``, the average over all pairs.
        """
        m = cor_spearman(self._u)
        return float(m[np.triu_indices(self._dim, 1)].mean())

    def beta(self) -> float:
        """Blomqvist's beta of the data: dependence measured at the medians.

        It compares how often all variables fall on the same side of their
        medians with what independence would give; between -1 and 1.

        Returns
        -------
        float
            The sample value from :func:`~rcopula.dependence.beta_n`.
        """
        return beta_n(self._u)

    def lambda_(self) -> TailDependence:
        r"""Estimate from the data how likely the two variables are to be extreme together.

        Tail dependence is the chance that one variable is extreme given that
        the other is equally extreme, in the limit; this gives rough estimates
        for the lower and upper tails. Two-dimensional data only.

        Returns
        -------
        TailDependence
            ``lower`` and ``upper`` estimates, nominally in ``[0, 1]``.

        Raises
        ------
        NotImplementedError
            If ``dim != 2``.

        Notes
        -----
        Uses the standard threshold estimators at :math:`p = n^{-1/2}`:
        :math:`\hat\lambda_L = C_n(p,p)/p` and
        :math:`\hat\lambda_U = (1 - 2(1-p) + C_n(1-p, 1-p))/p`.

        These converge slowly -- tail dependence is estimated from the handful
        of points in the corner, so treat them as indicative rather than
        precise.
        """
        if self._dim != 2:
            raise NotImplementedError("nonparametric tail dependence is implemented for dim=2 only")
        p = 1.0 / np.sqrt(self._n)
        lower = float(self.cdf([[p, p]])[0] / p)
        upper = float((1.0 - 2.0 * (1.0 - p) + self.cdf([[1.0 - p, 1.0 - p]])[0]) / p)
        return TailDependence(lower=lower, upper=upper)

    def describe(self) -> str:
        """Return a one-line summary: dimension, sample size and smoothing.

        Returns
        -------
        str
            E.g. ``"Empirical copula, dim 2, n=500, smoothing='none'"``.
        """
        return f"Empirical copula, dim {self._dim}, n={self._n}, smoothing={self.smoothing!r}"
