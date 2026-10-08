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
    """The result of fitting both the margins and the copula: a complete, usable distribution.

    Returned by :func:`fit_joint`; you do not normally build one yourself.
    The fitted :attr:`distribution` can be sampled or evaluated straight
    away, and :meth:`summary` prints a report.

    Parameters
    ----------
    distribution : CopulaDistribution
        Margins and copula, both fitted.
    copula : Copula
        The fitted dependence part alone.
    margin_params : list of tuple of float, length d
        Fitted parameters of each margin, in ``scipy`` order.
    loglik : float
        Joint log-likelihood, margins included.
    method : {"ifm", "ml"}
        Estimation method used.
    n_obs : int
        Number of observations (rows) fitted.
    converged : bool
        Whether the optimiser reported success (always ``True`` for
        ``"ifm"``).
    message : str, default ""
        Optimiser message (empty for ``"ifm"``).
    marginal_loglik : float, default 0.0
        The margins' share of ``loglik``.
    n_at_boundary : int, default 0
        Number of observations whose fitted probability integral transform
        landed on 0 or 1.
    margin_fixed : list of tuple of bool or None, default None
        For each margin, which entries of ``margin_params`` were pinned
        through ``margin_kwargs`` (``floc=0`` and the like) rather than
        estimated. ``None`` means nothing was pinned.
    _x : ndarray of float, shape (n, d)
        The data that were fitted. Internal; leave at its default.

    Attributes
    ----------
    distribution : CopulaDistribution
        Margins and copula, both fitted. Ready to ``rvs`` or ``pdf``.
    copula : Copula
        The dependence part alone.
    margin_params : list of tuple of float, length d
        The fitted parameters of each margin, in ``scipy`` order (shape
        parameters first, then ``loc`` and ``scale``).
    loglik : float
        Joint log-likelihood, margins included -- so it is **not** comparable
        with a copula-only log-likelihood from :func:`~rcopula.fit`.
    method : str
        ``"ifm"`` or ``"ml"``.
    n_obs : int
        Number of observations.
    converged : bool
        Whether the optimiser reported success.
    message : str
        Optimiser message, empty for ``"ifm"``.
    marginal_loglik : float
        Sum of the margins' log-densities at the fitted parameters.
    n_at_boundary : int
        Observations whose fitted probability integral transform landed exactly
        on 0 or 1. Almost always the sample extremes, because maximum likelihood
        puts a margin's support boundary there; those values are nudged just
        inside ``(0, 1)`` before the copula density is evaluated, and this
        count reports how many were patched.
    margin_fixed : list of tuple of bool or None
        Per margin, ``True`` where the parameter was pinned rather than
        estimated; ``None`` when nothing was pinned.
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
    margin_fixed: list[tuple[bool, ...]] | None = None
    _x: NDArray[np.float64] = field(repr=False, default_factory=lambda: np.empty((0, 0)))

    @property
    def n_params(self) -> int:
        """How many parameters were estimated in total, margins plus copula.

        Returns
        -------
        int
            The number of estimated margin parameters -- the lengths of
            ``margin_params`` minus any pinned through ``margin_kwargs`` (see
            :attr:`margin_fixed`) -- plus the number of free copula
            parameters. Pinned values were not estimated, so counting them
            would overstate AIC and BIC.
        """
        pinned = sum(sum(flags) for flags in self.margin_fixed) if self.margin_fixed else 0
        estimated = sum(len(p) for p in self.margin_params) - pinned
        return estimated + int(np.sum(self.copula.free))

    @property
    def aic(self) -> float:
        """A fit score for the whole model that penalises extra parameters; lower is better (AIC).

        Akaike information criterion, ``2 k - 2 loglik`` with ``k =``
        :attr:`n_params`.

        Returns
        -------
        float
        """
        return float(2 * self.n_params - 2 * self.loglik)

    @property
    def bic(self) -> float:
        """A fit score for the whole model with a stronger penalty than AIC; lower is better (BIC).

        Bayesian information criterion, ``k log n - 2 loglik`` with ``k =``
        :attr:`n_params`.

        Returns
        -------
        float
        """
        return float(self.n_params * np.log(self.n_obs) - 2 * self.loglik)

    @property
    def dependence_loglik(self) -> float:
        """The part of the log-likelihood due to the copula alone (joint minus margins).

        Use this, not :attr:`loglik`, to compare copula families.

        Returns
        -------
        float
            ``loglik - marginal_loglik``.

        Notes
        -----
        This *is* comparable across copula families fitted to the same margins,
        which the joint figure is not.
        """
        return float(self.loglik - self.marginal_loglik)

    def summary(self) -> str:
        """A human-readable text report of the fit, ready to print.

        Lists each margin's fitted parameters, the copula, the joint,
        marginal and dependence log-likelihoods, AIC/BIC, the parameter count
        and the number of boundary observations, plus a warning if the
        optimiser did not converge.

        Returns
        -------
        str
            Multi-line text; pass it to ``print``.

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


def _pinned(family: Any, n_params: int, kwargs: dict[str, Any]) -> tuple[bool, ...]:
    """Which of a scipy family's ``fit`` parameters ``kwargs`` holds fixed.

    Follows scipy's own conventions: ``f0``, ``f1``, ... or ``f<name>`` /
    ``fix_<name>`` for the shape parameters, in order, then ``floc`` and
    ``fscale``. Anything else in ``kwargs`` (a starting guess, an optimiser) pins
    nothing.
    """
    shapes_attr = getattr(family, "shapes", None) or ""
    shapes = shapes_attr.replace(",", " ").split()
    flags = [False] * n_params
    for j, name in enumerate(shapes[:n_params]):
        if any(key in kwargs for key in (f"f{j}", f"f{name}", f"fix_{name}")):
            flags[j] = True
    n_shapes = len(shapes)
    if "floc" in kwargs and n_shapes < n_params:
        flags[n_shapes] = True
    if "fscale" in kwargs and n_shapes + 1 < n_params:
        flags[n_shapes + 1] = True
    return tuple(flags)


def _fit_margins(
    distribution: CopulaDistribution,
    x: NDArray[np.float64],
    margin_kwargs: list[dict[str, Any]] | None,
) -> tuple[list[tuple[float, ...]], list[Any], list[tuple[bool, ...]]]:
    """Maximum likelihood for each margin separately.

    Also returns, per margin, which parameters ``margin_kwargs`` pinned.
    """
    fitted_params: list[tuple[float, ...]] = []
    frozen: list[Any] = []
    pinned: list[tuple[bool, ...]] = []
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
        pinned.append(_pinned(family, len(estimate), kwargs))
    return fitted_params, frozen, pinned


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
    """Fit a full joint distribution: each variable's own distribution plus the copula linking them.

    Use this when you want a complete model on the original scale -- to
    simulate realistic data, compute probabilities, or report fitted
    margins -- rather than the dependence alone. This is R's ``fitMvdc``.
    If you only care about the dependence, :func:`~rcopula.fit` on ranks is
    safer, because it makes no assumption about the margins.

    Parameters
    ----------
    distribution : CopulaDistribution
        Supplies the *shapes*: which copula family and which marginal families.
        Its current parameter values are only a starting point. Each margin
        must be a frozen ``scipy.stats`` distribution, so it can be refitted.
    x : array_like of float, shape (n, d)
        Data on the original scale, **not** pseudo-observations; ``d`` must
        equal ``distribution.dim``. A 1-D input is treated as a single row.
    method : {"ifm", "ml"}, default "ifm"
        ``"ifm"`` fits each margin first, then the copula (two steps, the
        standard choice). ``"ml"`` then optimises everything jointly, starting
        from the IFM answer. See the module docstring.
    margin_kwargs : list of dict or None, default None
        One dict per margin (length ``d``) of extra arguments for ``scipy``'s
        ``fit`` -- most usefully ``{"floc": 0}`` to pin a location that the
        family requires to be zero. Use ``{}`` for margins that need nothing.
        Parameters pinned this way (``f0``, ``f<shape>``, ``fix_<shape>``,
        ``floc``, ``fscale``) stay pinned under ``method="ml"`` too, and are
        not counted in :attr:`JointFitResult.n_params` (so not in AIC/BIC).
        Getting this wrong is the commonest cause of an implausible fit: a
        Gamma fitted with a free location will happily slide it to just below
        the sample minimum.
    copula_method : {"mpl", "ml", "itau", "irho", "itau.mpl"}, default "mpl"
        Passed to :func:`~rcopula.fit` for the copula step.

    Returns
    -------
    JointFitResult
        The fitted distribution, the copula, each margin's parameters, the
        log-likelihoods and AIC/BIC.

    Raises
    ------
    ValueError
        If ``x`` does not have ``distribution.dim`` columns, ``method`` is not
        ``"ifm"`` or ``"ml"``, or ``margin_kwargs`` has the wrong length.
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

    # Read a DataFrame the way CopulaDistribution does: by column name when the
    # distribution has names, so a reordered frame is not silently mismatched.
    x = np.atleast_2d(distribution._validate_x(x)) if hasattr(x, "columns") else x
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

    margin_params, frozen, margin_fixed = _fit_margins(distribution, x, margin_kwargs)
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
        # Margin parameters pinned through margin_kwargs (floc=0, ...) stay
        # pinned in the joint optimisation; only the estimated ones move.
        marginal_values = np.concatenate([np.asarray(p, dtype=float) for p in margin_params])
        marginal_free = ~np.concatenate([np.asarray(f, dtype=bool) for f in margin_fixed])
        sizes = [len(p) for p in margin_params]
        n_marginal_free = int(marginal_free.sum())
        start = np.concatenate([marginal_values[marginal_free], np.asarray(copula.params)[free]])

        def margin_vector(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            values = marginal_values.copy()
            values[marginal_free] = theta[:n_marginal_free]
            return values

        def unpack(theta: NDArray[np.float64]) -> CopulaDistribution | None:
            values = margin_vector(theta)
            position = 0
            candidates = []
            for j, size in enumerate(sizes):
                family = getattr(distribution.margins[j], "dist")  # noqa: B009
                try:
                    candidates.append(family(*values[position : position + size]))
                except (ValueError, TypeError):
                    return None
                position += size
            params = np.array(copula.params, dtype=float)
            params[free] = theta[n_marginal_free:]
            try:
                return CopulaDistribution(
                    copula.with_params(params), candidates, names=distribution.names
                )
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
            values = margin_vector(np.asarray(result.x))
            position = 0
            margin_params = []
            for size in sizes:
                margin_params.append(tuple(float(v) for v in values[position : position + size]))
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
        margin_fixed=margin_fixed if any(any(f) for f in margin_fixed) else None,
        _x=x,
    )
