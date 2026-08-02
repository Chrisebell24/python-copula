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

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from rcopula.core.base import Copula, TailDependence
from rcopula.dependence import beta_n, cor_kendall, cor_spearman, pseudo_obs

__all__ = ["EmpiricalCopula"]

SMOOTHINGS = ("none", "beta", "checkerboard")


class EmpiricalCopula(Copula):
    r"""Nonparametric copula estimated from data.

    Parameters
    ----------
    data : array_like
        ``(n, d)`` observations. Converted to pseudo-observations internally, so
        raw data on any scale is fine.
    smoothing : {"none", "beta", "checkerboard"}
        ``"none"`` is the classical step-function estimator. ``"beta"`` and
        ``"checkerboard"`` are smoothed and are genuine copulas; only ``"beta"``
        admits a density.
    offset : float
        Added to the denominator, as in R's ``empCopula``. Rarely needed.
    ties_method : str
        Passed to :func:`~rcopula.dependence.pseudo_obs`.

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
        **kwargs: object,
    ) -> None:
        if smoothing not in SMOOTHINGS:
            raise ValueError(f"smoothing must be one of {SMOOTHINGS}, got {smoothing!r}")

        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] < 2:
            raise ValueError("an empirical copula needs at least two observations")

        self.smoothing = smoothing
        self.offset = float(offset)
        self.ties_method = ties_method
        self._u = np.asarray(pseudo_obs(arr, ties_method=ties_method), dtype=np.float64)
        self._n = arr.shape[0]
        # Ranks in 1..n, which the smoothed estimators are parameterised by.
        self._ranks = np.round(self._u * (self._n + 1.0)).astype(int)

        super().__init__(np.empty(0), arr.shape[1])

    # -- properties -----------------------------------------------------

    @property
    def n_obs(self) -> int:
        """Number of observations the estimator was built from."""
        return self._n

    @property
    def pseudo_observations(self) -> NDArray[np.float64]:
        """The rank-transformed data underlying the estimator."""
        return self._u

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
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

        if self.smoothing == "beta":
            # C_n^beta(u) = (1/n) sum_i prod_j pbeta(u_j; R_ij, n - R_ij + 1)
            out = np.empty(u.shape[0])
            for k, point in enumerate(u):
                terms = stats.beta.cdf(point[None, :], self._ranks, n - self._ranks + 1)
                out[k] = np.prod(terms, axis=1).mean()
            return out

        # Checkerboard: each observation is spread over one grid cell.
        out = np.empty(u.shape[0])
        for k, point in enumerate(u):
            terms = np.clip(n * point[None, :] - self._ranks + 1.0, 0.0, 1.0)
            out[k] = np.prod(terms, axis=1).mean()
        return out

    def _logpdf(self, u, params):
        if self.smoothing != "beta":
            raise NotImplementedError(
                "the unsmoothed empirical copula is a step function and the "
                "checkerboard estimator is piecewise uniform, so neither has a "
                "density; use smoothing='beta' if you need one"
            )
        n = self._n
        out = np.empty(u.shape[0])
        for k, point in enumerate(u):
            terms = stats.beta.pdf(point[None, :], self._ranks, n - self._ranks + 1)
            out[k] = np.prod(terms, axis=1).mean()
        with np.errstate(divide="ignore"):
            return np.log(out)

    def _rvs(self, size, params, rng):
        """Resample the pseudo-observations, smoothing if requested."""
        idx = rng.integers(0, self._n, size=size)
        if self.smoothing == "none":
            return self._u[idx]
        if self.smoothing == "beta":
            r = self._ranks[idx]
            return rng.beta(r, self._n - r + 1)
        # Checkerboard: uniform within the selected cell.
        return (self._ranks[idx] - rng.uniform(size=(size, self._dim))) / self._n

    # -- estimators R exposes as free functions -------------------------

    def dCdu(self, u: ArrayLike, bandwidth: float | None = None) -> NDArray[np.float64]:
        r"""Partial derivatives :math:`\partial C_n/\partial u_j` (R's ``dCn``).

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
        """Sample Kendall's tau (averaged over pairs when ``dim > 2``)."""
        m = cor_kendall(self._u)
        return float(m[np.triu_indices(self._dim, 1)].mean())

    def rho(self) -> float:
        """Sample Spearman's rho (averaged over pairs when ``dim > 2``)."""
        m = cor_spearman(self._u)
        return float(m[np.triu_indices(self._dim, 1)].mean())

    def beta(self) -> float:
        """Sample Blomqvist's beta."""
        return beta_n(self._u)

    def lambda_(self) -> TailDependence:
        r"""Nonparametric tail-dependence estimates.

        Uses the standard threshold estimators at :math:`p = n^{-1/2}`:
        :math:`\hat\lambda_L = C_n(p,p)/p` and
        :math:`\hat\lambda_U = (1 - 2p + C_n(1-p, 1-p))/p`.

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
        return f"Empirical copula, dim {self._dim}, n={self._n}, smoothing={self.smoothing!r}"
