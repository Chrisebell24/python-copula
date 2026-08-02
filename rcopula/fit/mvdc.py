r"""Fitting margins and copula together (R's ``fitMvdc``).

Everywhere else in this package the margins are removed first, by
:func:`~rcopula.pseudo_obs`, and only the copula is estimated. That is the
semiparametric route and usually the right one: it makes no claim about the
marginal shapes, so a wrong guess there cannot contaminate the dependence.

Sometimes you want the whole distribution anyway -- to simulate from it, to
price something, to report a fitted model rather than a fitted copula. That
means committing to parametric margins, and there are two ways to do it.

``"ifm"`` -- **inference functions for margins** (Joe and Xu 1996). Fit each
margin by maximum likelihood, then fit the copula to
:math:`(\hat F_1(x_1), \dots, \hat F_d(x_d))`. Two steps, each small, and the
standard choice. Note what changed: the copula now sees *parametric* probability
integral transforms, not ranks, so a misspecified margin distorts the estimated
dependence -- which the rank-based route would have been immune to.

``"ml"`` -- **full maximum likelihood**. Optimise every parameter at once.
Asymptotically efficient, and in practice often not worth it: the surface has
:math:`\sum_j p_j + q` dimensions, it is not concave, and the gain over IFM is
usually in the third decimal. Started from the IFM estimate, because started
anywhere else it frequently does not arrive.

============================  ================================================
:func:`fit_joint`             Estimate margins and copula from data.
:class:`JointFitResult`       The fitted distribution, with both parts.
============================  ================================================

Examples
--------
>>> import numpy as np, rcopula as rc
>>> from scipy import stats
>>> from rcopula.fit.mvdc import fit_joint
>>> truth = rc.CopulaDistribution(
...     rc.ClaytonCopula(2.0), [stats.norm(1.0, 2.0), stats.expon(scale=3.0)]
... )
>>> x = truth.rvs(3000, random_state=0)
>>> template = rc.CopulaDistribution(rc.ClaytonCopula(1.0), [stats.norm(), stats.expon()])
>>> result = fit_joint(template, x)
>>> bool(abs(result.copula.params[0] - 2.0) < 0.25)
True

References
----------
Joe, H. and Xu, J. J. (1996). The estimation method of inference functions for
    margins for multivariate models. Technical Report 166, Department of
    Statistics, University of British Columbia.
    The IFM estimator.
Joe, H. (2005). Asymptotic efficiency of the two-stage estimation method for
    copula-based models. *J. Multivariate Analysis* 94(2), 401-419.
    How much IFM gives up against full maximum likelihood, which is usually
    very little.
Genest, C., Ghoudi, K. and Rivest, L.-P. (1995). A semiparametric estimation
    procedure of dependence parameters in multivariate families of
    distributions. *Biometrika* 82(3), 543-552.
    The rank-based alternative, and why it is the safer default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

from rcopula.core.base import Copula
from rcopula.distribution import CopulaDistribution

__all__ = ["JointFitResult", "fit_joint"]

Method = Literal["ifm", "ml"]

#: Log-density floor. A candidate parameter vector can put an observation
#: outside a margin's support, where the density is genuinely zero; that must
#: cost the optimiser a lot without producing an infinity it cannot compare.
_LOG_FLOOR = -1e6


@dataclass
class JointFitResult:
    """A fitted joint distribution.

    Attributes
    ----------
    distribution : CopulaDistribution
        Margins and copula, both fitted. Ready to ``rvs`` or ``pdf``.
    copula : Copula
        The dependence part alone.
    margin_params : list of tuple
        The fitted parameters of each margin, in ``scipy`` order (shape
        parameters first, then ``loc`` and ``scale``).
    loglik : float
        Joint log-likelihood, margins included -- so it is **not** comparable
        with a copula-only log-likelihood from :func:`~rcopula.fit`.
    method : str
    n_obs : int
    converged : bool
    n_at_boundary : int
        Observations whose fitted probability integral transform landed exactly
        on 0 or 1. Almost always the sample extremes, because maximum likelihood
        puts a margin's support boundary there; see :func:`_joint_loglik`.
    """

    distribution: CopulaDistribution
    copula: Copula
    margin_params: list[tuple[float, ...]]
    loglik: float
    method: str
    n_obs: int
    converged: bool
    message: str = ""
    marginal_loglik: float = 0.0
    n_at_boundary: int = 0
    _x: NDArray[np.float64] = field(repr=False, default_factory=lambda: np.empty((0, 0)))

    @property
    def n_params(self) -> int:
        """Every parameter estimated, marginal and dependence."""
        return sum(len(p) for p in self.margin_params) + int(np.sum(self.copula.free))

    @property
    def aic(self) -> float:
        """Akaike information criterion for the joint model."""
        return float(2 * self.n_params - 2 * self.loglik)

    @property
    def bic(self) -> float:
        """Bayesian information criterion for the joint model."""
        return float(self.n_params * np.log(self.n_obs) - 2 * self.loglik)

    @property
    def dependence_loglik(self) -> float:
        """The copula's contribution: joint log-likelihood minus the margins'.

        This *is* comparable across copula families fitted to the same margins,
        which the joint figure is not.
        """
        return float(self.loglik - self.marginal_loglik)

    def summary(self) -> str:
        """A printable report.

        Examples
        --------
        >>> import rcopula as rc
        >>> from scipy import stats
        >>> from rcopula.fit.mvdc import fit_joint
        >>> truth = rc.CopulaDistribution(rc.GumbelCopula(2.0), [stats.norm()] * 2)
        >>> x = truth.rvs(500, random_state=0)
        >>> template = rc.CopulaDistribution(rc.GumbelCopula(1.5), [stats.norm()] * 2)
        >>> print(fit_joint(template, x).summary().splitlines()[0])
        Joint fit by IFM, 500 observations
        """
        lines = [
            f"Joint fit by {self.method.upper()}, {self.n_obs} observations",
            "=" * 68,
        ]
        for j, params in enumerate(self.margin_params):
            name = getattr(getattr(self.distribution.margins[j], "dist", None), "name", "margin")
            values = ", ".join(f"{p:.6f}" for p in params)
            lines.append(f"  margin {j} ({name:<12}) {values}")
        lines += [
            "",
            "  copula               " + self.copula.describe(),
            "",
            f"  joint log-lik        {self.loglik: .4f}",
            f"  of which margins     {self.marginal_loglik: .4f}",
            f"  of which dependence  {self.dependence_loglik: .4f}",
            f"  AIC / BIC            {self.aic: .4f} / {self.bic:.4f}",
            f"  parameters           {self.n_params}",
            "",
            f"  at a margin boundary {self.n_at_boundary} observation(s)",
            "",
            "  The joint log-likelihood includes the margins, so it is not",
            "  comparable with a copula-only one. Compare dependence_loglik",
            "  across families instead, and only at identical margins.",
        ]
        if not self.converged:
            lines += ["", f"  WARNING: optimiser did not converge -- {self.message}"]
        return "\n".join(lines)


def _fit_margins(
    distribution: CopulaDistribution,
    x: NDArray[np.float64],
    margin_kwargs: list[dict[str, Any]] | None,
) -> tuple[list[tuple[float, ...]], list[Any]]:
    """Maximum likelihood for each margin separately."""
    fitted_params: list[tuple[float, ...]] = []
    frozen: list[Any] = []
    for j, margin in enumerate(distribution.margins):
        family: Any = getattr(margin, "dist", None)
        if family is None or not hasattr(family, "fit"):
            raise TypeError(
                f"margin {j} ({type(margin).__name__}) cannot be refitted: it has "
                "no underlying scipy distribution with a .fit method. Fit it "
                "yourself and pass the frozen result, then use rcopula.fit on "
                "the pseudo-observations."
            )
        kwargs = (margin_kwargs[j] if margin_kwargs else {}) or {}
        estimate = tuple(float(p) for p in family.fit(x[:, j], **kwargs))
        fitted_params.append(estimate)
        frozen.append(family(*estimate))
    return fitted_params, frozen


def _joint_loglik(copula: Copula, margins: list[Any], x: NDArray[np.float64]) -> tuple[float, int]:
    r"""Joint log-likelihood, and how many observations needed rescuing.

    Fitting a margin by maximum likelihood usually places its support boundary
    *at* the sample extreme -- ``scipy``'s exponential sets ``loc`` to the
    minimum -- so the smallest observation maps to :math:`F(x) = 0` exactly, and
    the copula density there is undefined. That is a property of the estimator,
    not of the data, and it would otherwise report an infinite log-likelihood for
    a perfectly good fit.

    The probability integral transforms are therefore nudged inside the open
    cube, and the count of affected observations is returned so the caller can
    say how much was patched rather than hiding it.
    """
    u = np.column_stack([np.asarray(m.cdf(x[:, j]), dtype=float) for j, m in enumerate(margins)])
    touched = int(np.sum(np.any((u <= 0.0) | (u >= 1.0), axis=1)))
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.asarray(copula.logpdf(u), dtype=float)
        for j, margin in enumerate(margins):
            values = values + np.log(np.asarray(margin.pdf(x[:, j]), dtype=float))
    values = np.where(np.isfinite(values), values, _LOG_FLOOR)
    return float(np.sum(values)), touched


def _marginal_loglik(margins: list[Any], x: NDArray[np.float64]) -> float:
    total = 0.0
    for j, margin in enumerate(margins):
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.log(np.asarray(margin.pdf(x[:, j]), dtype=float))
        total += float(np.sum(np.where(np.isfinite(values), values, _LOG_FLOOR)))
    return total


def fit_joint(
    distribution: CopulaDistribution,
    x: ArrayLike,
    *,
    method: Method = "ifm",
    margin_kwargs: list[dict[str, Any]] | None = None,
    copula_method: str = "mpl",
) -> JointFitResult:
    """Fit a copula and its margins to data (R's ``fitMvdc``).

    Parameters
    ----------
    distribution : CopulaDistribution
        Supplies the *shapes*: which copula family and which marginal families.
        Its current parameter values are only a starting point.
    x : array_like, shape (n, d)
        Data on the original scale, **not** pseudo-observations.
    method : {"ifm", "ml"}
        Two-step or joint. See the module docstring.
    margin_kwargs : list of dict, optional
        Extra arguments per margin for ``scipy``'s ``fit`` -- most usefully
        ``{"floc": 0}`` to pin a location that the family requires to be zero.
        Getting this wrong is the commonest cause of an implausible fit: a
        Gamma fitted with a free location will happily slide it to just below
        the sample minimum.
    copula_method : str
        Passed to :func:`~rcopula.fit` for the copula step.

    Returns
    -------
    JointFitResult

    Raises
    ------
    TypeError
        If a margin has no underlying ``scipy`` distribution to refit.

    Notes
    -----
    ``"ml"`` starts from the ``"ifm"`` estimate. That is not a convenience: the
    joint surface is not concave, and from a cold start the optimiser often ends
    somewhere that is not the maximum at all.

    Examples
    --------
    >>> import numpy as np, rcopula as rc
    >>> from scipy import stats
    >>> from rcopula.fit.mvdc import fit_joint
    >>> truth = rc.CopulaDistribution(
    ...     rc.GumbelCopula(2.5), [stats.norm(1.0, 2.0), stats.norm(-1.0, 0.5)]
    ... )
    >>> x = truth.rvs(4000, random_state=0)
    >>> template = rc.CopulaDistribution(rc.GumbelCopula(1.5), [stats.norm()] * 2)
    >>> result = fit_joint(template, x)
    >>> bool(abs(result.margin_params[0][0] - 1.0) < 0.1)
    True
    >>> bool(abs(result.copula.params[0] - 2.5) < 0.2)
    True

    The fitted object is a distribution, so it can be sampled straight away:

    >>> result.distribution.rvs(5, random_state=0).shape
    (5, 2)
    """
    from rcopula.fit.api import fit as fit_copula

    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[1] != distribution.dim:
        raise ValueError(
            f"x has {x.shape[1]} columns but the distribution has dim={distribution.dim}"
        )
    if method not in ("ifm", "ml"):
        raise ValueError(f"method must be 'ifm' or 'ml', got {method!r}")
    if margin_kwargs is not None and len(margin_kwargs) != distribution.dim:
        raise ValueError(
            f"margin_kwargs must have {distribution.dim} entries, got {len(margin_kwargs)}"
        )

    margin_params, frozen = _fit_margins(distribution, x, margin_kwargs)
    u = np.clip(
        np.column_stack([np.asarray(m.cdf(x[:, j])) for j, m in enumerate(frozen)]),
        1e-10,
        1 - 1e-10,
    )
    copula_fit = fit_copula(distribution.copula, u, method=copula_method)
    copula = copula_fit.copula

    fitted = CopulaDistribution(copula, frozen, names=distribution.names)
    marginal = _marginal_loglik(frozen, x)
    loglik, boundary = _joint_loglik(copula, frozen, x)
    converged, message = True, ""

    if method == "ml":
        free = np.asarray(copula.free, dtype=bool)
        sizes = [len(p) for p in margin_params]
        marginal_start = np.concatenate([np.asarray(p) for p in margin_params])
        start = np.concatenate([marginal_start, np.asarray(copula.params)[free]])

        def unpack(theta: NDArray[np.float64]) -> CopulaDistribution | None:
            position = 0
            candidates = []
            for j, size in enumerate(sizes):
                family = getattr(distribution.margins[j], "dist")  # noqa: B009
                try:
                    candidates.append(family(*theta[position : position + size]))
                except (ValueError, TypeError):
                    return None
                position += size
            params = np.array(copula.params, dtype=float)
            params[free] = theta[position:]
            try:
                return CopulaDistribution(copula.with_params(params), candidates)
            except (ValueError, np.linalg.LinAlgError):
                return None

        def objective(theta: NDArray[np.float64]) -> float:
            candidate = unpack(theta)
            if candidate is None:
                return 1e12
            return float(-_joint_loglik(candidate.copula, list(candidate.margins), x)[0])

        result = optimize.minimize(
            objective, start, method="Nelder-Mead", options={"maxiter": 4000, "fatol": 1e-8}
        )
        best = unpack(np.asarray(result.x))
        if best is not None and -result.fun > loglik:
            fitted = best
            copula = best.copula
            position = 0
            margin_params = []
            for size in sizes:
                margin_params.append(tuple(float(v) for v in result.x[position : position + size]))
                position += size
            marginal = _marginal_loglik(list(best.margins), x)
            loglik, boundary = _joint_loglik(best.copula, list(best.margins), x)
        converged, message = bool(result.success), str(result.message)

    return JointFitResult(
        distribution=fitted,
        copula=copula,
        margin_params=margin_params,
        loglik=loglik,
        method=method,
        n_obs=int(x.shape[0]),
        converged=converged,
        message=message,
        marginal_loglik=marginal,
        n_at_boundary=boundary,
        _x=x,
    )
