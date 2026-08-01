r"""Time-varying copulas: dependence that moves.

Every copula elsewhere in this package is *constant* -- one parameter for the
whole sample. That is often indefensible. Dependence between assets rises in a
crisis and falls afterwards; dependence between a river's peak and its volume
changes with land use; dependence between two sensors changes with the weather.
Fitting one number to all of it produces an average that describes no particular
period.

The fix is to let the copula parameter follow an observation-driven recursion,
so :math:`\theta_t` is a deterministic function of the past. The likelihood
stays exact -- there is no filtering approximation and no latent state to
integrate out -- which is the whole appeal of the observation-driven class.

Two recursions are provided.

**Patton (2006).** The parameter responds to a moving average of a *forcing*
term computed from the last :math:`q` observations:

.. math::

    \theta_t = \Lambda\!\left(\omega + \beta\,\theta_{t-1}
               + \alpha\, \frac{1}{q}\sum_{j=1}^{q} m(u_{t-j}, v_{t-j})\right),

with :math:`\Lambda` a link keeping :math:`\theta_t` inside its domain. Patton
used :math:`m = \Phi^{-1}(u)\Phi^{-1}(v)` for correlation parameters and
:math:`m = |u - v|` otherwise. The second is *decreasing* in dependence, so a
fitted :math:`\alpha` is normally negative for it -- a sign flip that is easy to
misread as a bug.

**GAS (Creal, Koopman and Lucas, 2013).** The parameter responds to the *score*
of its own likelihood, which is the direction that would most improve the fit at
the last observation:

.. math::

    f_{t+1} = \omega + \beta f_t + \alpha s_t, \qquad
    s_t = \frac{\partial \log c(u_t, v_t; \Lambda(f_t))}{\partial f_t},
    \qquad \theta_t = \Lambda(f_t).

The score is family-specific and automatic: no forcing term has to be chosen,
and a tail observation moves a Clayton copula differently from a Gumbel one.
This usually fits better than Patton's recursion and is harder to explain, which
is a fair summary of the trade-off between them.

For :math:`d > 2` elliptical copulas the natural object is a whole correlation
*matrix*, and the standard recursion for that is Engle's DCC -- see
:func:`fit_dcc`.

============================  ================================================
:class:`DynamicCopula`        A family plus a parameter recursion.
:func:`fit_dynamic`           Estimate the recursion by maximum likelihood.
:class:`DynamicFitResult`     What came back, with a test against constancy.
:func:`fit_dcc`               Engle's DCC for a time-varying correlation matrix.
:class:`DccResult`            The filtered correlation path.
============================  ================================================

.. warning::

   A dynamic copula is fitted to **pseudo-observations of i.i.d. innovations**,
   not to raw data. Volatility clustering left in the margins will be picked up
   by this recursion and reported as time-varying dependence, which is precisely
   the confusion :mod:`rcopula.garch` exists to prevent. Filter the margins
   first.

Examples
--------
>>> import numpy as np, rcopula as rc
>>> from rcopula.dynamic import DynamicCopula
>>> model = DynamicCopula(rc.GaussianCopula(0.0), coefficients=(0.02, 0.1, 0.9))
>>> u = model.simulate(500, random_state=0)
>>> path = model.filter(u).path
>>> float(path.min()) > -1.0 and float(path.max()) < 1.0
True

References
----------
Patton, A. J. (2006). Modelling asymmetric exchange rate dependence.
    *International Economic Review* 47(2), 527-556.
Creal, D., Koopman, S. J. and Lucas, A. (2013). Generalized autoregressive
    score models with applications. *Journal of Applied Econometrics*
    28(5), 777-795.
Engle, R. F. (2002). Dynamic conditional correlation: a simple class of
    multivariate generalized autoregressive conditional heteroskedasticity
    models. *Journal of Business and Economic Statistics* 20(3), 339-350.
Aielli, G. P. (2013). Dynamic conditional correlation: on properties and
    estimation. *Journal of Business and Economic Statistics* 31(3), 282-299.
    Why the two-step DCC estimator is inconsistent, and the cDCC correction.
Manner, H. and Reznikova, O. (2012). A survey on time-varying copulas:
    specification, simulations, and application.
    *Econometric Reviews* 31(6), 654-687.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, optimize, special, stats

from rcopula.core.base import Copula
from rcopula.core.elliptical import GaussianCopula, StudentCopula
from rcopula.dependence import pseudo_obs

__all__ = [
    "DccResult",
    "DynamicCopula",
    "DynamicFitResult",
    "DynamicPath",
    "fit_dcc",
    "fit_dynamic",
]

Driver = Literal["patton", "gas"]
Forcing = Literal["auto", "normal-product", "abs-difference"]

#: How far a one-sided parameter is allowed to run. Clayton at 25 is tau = 0.93
#: and Gumbel at 25 is tau = 0.96; past that the copula is comonotone for any
#: practical purpose and the likelihood is flat, so a bound here costs nothing
#: and keeps the link from saturating. Override with ``bounds=``.
_ONE_SIDED_SPAN = 25.0

#: Frank's parameter is unbounded in both directions but numerically dead past
#: about 40 in either direction.
_TWO_SIDED_SPAN = 40.0

#: Quantiles are clipped before the normal-product forcing term, which otherwise
#: takes infinite values at a pseudo-observation of exactly 0 or 1.
_QUANTILE_CLIP = 1e-6

#: Step for the finite-difference score in the GAS recursion, on the linked
#: scale. Central differences, so the error is O(h^2) ~ 1e-8 -- far below the
#: Monte Carlo noise in anything this is used for.
_SCORE_STEP = 1e-4


# --------------------------------------------------------------------------
# links
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Link:
    """A smooth bijection from the real line onto ``(lower, upper)``.

    The recursion runs unconstrained and the link maps back, which is what
    stops an optimiser from ever proposing a Clayton with a negative parameter.
    """

    lower: float
    upper: float

    def __call__(self, x: NDArray[np.float64] | float) -> Any:
        # Written through tanh rather than the logistic so that it stays exact
        # for large |x| instead of overflowing in exp.
        value = self.centre + self.half * np.tanh(0.5 * np.asarray(x, dtype=float))
        # tanh returns exactly 1.0 for |x| past about 19, which would hand a
        # correlation of exactly 1 to the family and fail its Cholesky. The
        # endpoints are open, so the clip is restoring the link's own contract.
        margin = 1e-9 * self.half
        return np.clip(value, self.lower + margin, self.upper - margin)

    def inverse(self, theta: float) -> float:
        z = np.clip((theta - self.centre) / self.half, -1 + 1e-12, 1 - 1e-12)
        return float(2.0 * np.arctanh(z))

    def standardise(self, theta: float) -> float:
        """Map the parameter onto ``(-1, 1)``.

        Patton writes his recursion with ``beta`` multiplying the *lagged
        parameter itself*, which works because every parameter he applies it to
        already lives on a bounded, order-one scale -- a correlation, or a tail
        dependence coefficient. For a family like Clayton, whose parameter runs
        to 25, the same equation saturates the link on the first step.

        Standardising first makes ``beta`` dimensionless and is the identity
        exactly when the range is ``(-1, 1)``, so Patton's own case is
        reproduced unchanged.
        """
        return float((theta - self.centre) / self.half)

    @property
    def centre(self) -> float:
        return 0.5 * (self.lower + self.upper)

    @property
    def half(self) -> float:
        return 0.5 * (self.upper - self.lower)


def _default_link(copula: Copula, index: int, bounds: tuple[float, float] | None) -> _Link:
    """Pick a working range for the varying parameter.

    ``param_bounds`` is the family's mathematical domain, which is often
    infinite. A recursion needs a finite one, so an unbounded side is replaced
    by a span past which the family is numerically indistinguishable from its
    limit.
    """
    if bounds is not None:
        lower, upper = float(bounds[0]), float(bounds[1])
        if not lower < upper:
            raise ValueError(f"bounds must be increasing, got {bounds!r}")
        return _Link(lower, upper)

    lower, upper = copula.param_bounds[index]
    if np.isfinite(lower) and np.isfinite(upper):
        return _Link(float(lower), float(upper))
    if np.isfinite(lower):
        return _Link(float(lower), float(lower) + _ONE_SIDED_SPAN)
    if np.isfinite(upper):
        return _Link(float(upper) - _ONE_SIDED_SPAN, float(upper))
    return _Link(-_TWO_SIDED_SPAN, _TWO_SIDED_SPAN)


# --------------------------------------------------------------------------
# forcing terms
# --------------------------------------------------------------------------


def _forcing_values(u: NDArray[np.float64], forcing: str, df: float | None) -> NDArray[np.float64]:
    """The per-observation term, before it is averaged over the lag window."""
    if forcing == "normal-product":
        clipped = np.clip(u, _QUANTILE_CLIP, 1 - _QUANTILE_CLIP)
        # Patton uses the copula's own marginal quantile, so a t copula gets
        # t quantiles -- which matters, because they are much heavier-tailed
        # and the forcing term is a product of two of them.
        quantile = stats.norm.ppf if df is None else stats.t(df).ppf
        z = quantile(clipped)
        return np.asarray(np.prod(z, axis=1), dtype=float)
    if forcing == "abs-difference":
        return np.asarray(np.abs(u[:, 0] - u[:, 1]), dtype=float)
    raise ValueError(f"unknown forcing {forcing!r}")


def _moving_average(values: NDArray[np.float64], lags: int) -> NDArray[np.float64]:
    """Mean of the previous ``lags`` values, with the pre-sample gap filled by
    the sample mean.

    The alternative -- dropping the first ``lags`` observations -- changes the
    sample as ``lags`` changes and makes two fits incomparable.
    """
    n = values.size
    if lags < 1:
        raise ValueError(f"lags must be at least 1, got {lags}")
    cumulative = np.concatenate([[0.0], np.cumsum(values)])
    out = np.empty(n, dtype=float)
    usable = np.arange(lags, n)
    out[usable] = (cumulative[usable] - cumulative[usable - lags]) / lags
    out[:lags] = float(np.mean(values)) if n else 0.0
    return out


# --------------------------------------------------------------------------
# vectorised densities with a parameter that changes every row
# --------------------------------------------------------------------------
#
# The generic path evaluates ``family._logpdf`` once per observation, because
# the parameter differs at every one. That costs about 50 microseconds a call --
# fine to evaluate a model once, far too slow inside an optimiser.
#
# For the five families that are actually used dynamically in the literature the
# bivariate log-density is one vectorised expression, which is roughly two
# hundred times faster. Each is checked against the generic loop in
# ``tests/test_dynamic.py``; if one ever drifts from the family it mirrors, that
# test fails rather than the answer quietly changing.
#
# A fast path returns ``None`` when any parameter in the path leaves the region
# the expression is valid on, and the generic loop takes over for that call. It
# is never a question of accuracy, only of speed.


# Each family contributes a *pair*: a ``prepare`` step that depends only on the
# data, and a ``density`` step that depends on the parameter. The split matters
# for the GAS recursion, which is sequential and evaluates the density three
# times per observation -- with the split, the marginal quantile transform is
# paid once for the whole sample instead of 3n times.


def _prepare_normal(u: NDArray[np.float64], _: Any) -> NDArray[np.float64]:
    return np.asarray(stats.norm.ppf(np.clip(u, _QUANTILE_CLIP, 1 - _QUANTILE_CLIP)))


def _prepare_student(u: NDArray[np.float64], df: Any) -> NDArray[np.float64]:
    if df is None:
        raise ValueError("a Student copula needs degrees of freedom")
    return np.asarray(stats.t(float(df)).ppf(np.clip(u, _QUANTILE_CLIP, 1 - _QUANTILE_CLIP)))


def _prepare_log(u: NDArray[np.float64], _: Any) -> NDArray[np.float64]:
    return np.log(u)


def _prepare_gumbel(u: NDArray[np.float64], _: Any) -> NDArray[np.float64]:
    x = -np.log(u)
    return np.column_stack([x, np.log(x)])


def _prepare_identity(u: NDArray[np.float64], _: Any) -> NDArray[np.float64]:
    return u


def _fast_gaussian(z: NDArray[np.float64], theta: NDArray[np.float64], _: Any) -> Any:
    if np.any(np.abs(theta) >= 1.0):
        return None
    x, y = z[:, 0], z[:, 1]
    one_minus = 1.0 - theta**2
    return -0.5 * np.log(one_minus) - (theta**2 * (x**2 + y**2) - 2 * theta * x * y) / (
        2.0 * one_minus
    )


def _fast_student(z: NDArray[np.float64], theta: NDArray[np.float64], df: Any) -> Any:
    if df is None or np.any(np.abs(theta) >= 1.0):
        return None
    nu = float(df)
    x, y = z[:, 0], z[:, 1]
    one_minus = 1.0 - theta**2
    constant = (
        special.gammaln((nu + 2) / 2) + special.gammaln(nu / 2) - 2 * special.gammaln((nu + 1) / 2)
    )
    quadratic = (x**2 - 2 * theta * x * y + y**2) / (nu * one_minus)
    return (
        constant
        - 0.5 * np.log(one_minus)
        - (nu + 2) / 2 * np.log1p(quadratic)
        + (nu + 1) / 2 * (np.log1p(x**2 / nu) + np.log1p(y**2 / nu))
    )


def _fast_clayton(logu: NDArray[np.float64], theta: NDArray[np.float64], _: Any) -> Any:
    # Negative theta puts part of the unit square outside the support, where the
    # density is zero and the expression below is not defined. Hand those back.
    if np.any(theta <= 1e-8):
        return None
    x, y = logu[:, 0], logu[:, 1]
    inner = np.exp(-theta * x) + np.exp(-theta * y) - 1.0
    if np.any(inner <= 0):
        return None
    return np.log1p(theta) - (1.0 + theta) * (x + y) - (2.0 + 1.0 / theta) * np.log(inner)


def _fast_gumbel(prepared: NDArray[np.float64], theta: NDArray[np.float64], _: Any) -> Any:
    if np.any(theta < 1.0):
        return None
    x, y, log_x, log_y = (prepared[:, 0], prepared[:, 1], prepared[:, 2], prepared[:, 3])
    # s = (x^theta + y^theta)^(1/theta), through a log-sum-exp so that x^theta
    # cannot overflow at theta near the top of the link's range.
    log_s = special.logsumexp(np.column_stack([theta * log_x, theta * log_y]), axis=1) / theta
    s = np.exp(log_s)
    return (
        -s
        + (theta - 1.0) * (log_x + log_y)
        + x
        + y
        + (1.0 - 2.0 * theta) * log_s
        + np.log(s + theta - 1.0)
    )


def _fast_frank(u: NDArray[np.float64], theta: NDArray[np.float64], _: Any) -> Any:
    if np.any(np.abs(theta) < 1e-6):
        return None  # the theta -> 0 limit is independence; let the family say so
    # The denominator (1-e^-t) - (1-e^-tu)(1-e^-tv) expands exactly to
    # e^-tu + e^-tv - e^-t(u+v) - e^-t. Written that way it is four exponentials
    # instead of a difference of two numbers that both approach 1 as theta
    # grows -- which costs eleven digits at theta = 34 if left alone.
    exponents = -theta[:, None] * np.column_stack(
        [u[:, 0], u[:, 1], u[:, 0] + u[:, 1], np.ones(u.shape[0])]
    )
    signs = np.broadcast_to(np.array([1.0, 1.0, -1.0, -1.0]), exponents.shape)
    log_denominator, denominator_sign = special.logsumexp(
        exponents, b=signs, axis=1, return_sign=True
    )
    if np.any(denominator_sign == 0):
        return None
    return (
        np.log(np.abs(theta))
        + np.log(np.abs(np.expm1(-theta)))
        - theta * (u[:, 0] + u[:, 1])
        - 2.0 * log_denominator
    )


#: Family class -> (prepare, density). See the note above.
_FAST_LOGPDF: dict[str, tuple[Any, Any]] = {
    "GaussianCopula": (_prepare_normal, _fast_gaussian),
    "StudentCopula": (_prepare_student, _fast_student),
    "ClaytonCopula": (_prepare_log, _fast_clayton),
    "GumbelCopula": (_prepare_gumbel, _fast_gumbel),
    "FrankCopula": (_prepare_identity, _fast_frank),
}


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DynamicPath:
    """The filtered output of a :class:`DynamicCopula`.

    Attributes
    ----------
    path : ndarray, shape (n,)
        The copula parameter at each observation, on its natural scale.
    linked : ndarray, shape (n,)
        The same path before the link -- the scale the recursion runs on.
    loglik_contributions : ndarray, shape (n,)
        ``log c(u_t; theta_t)`` term by term. Summing gives the log-likelihood;
        the series itself shows which observations the model finds surprising.
    """

    path: NDArray[np.float64]
    linked: NDArray[np.float64]
    loglik_contributions: NDArray[np.float64]

    @property
    def loglik(self) -> float:
        """Total log-likelihood."""
        return float(np.sum(self.loglik_contributions))


class DynamicCopula:
    r"""A bivariate copula whose parameter follows an observation-driven recursion.

    Parameters
    ----------
    family : Copula
        A bivariate copula. Its parameter values set the starting point of the
        recursion and, for a multi-parameter family such as
        :class:`~rcopula.StudentCopula`, fix everything that does not vary.
    coefficients : sequence of 3 floats
        ``(omega, alpha, beta)``.
    driver : {"patton", "gas"}
        Which recursion. ``"patton"`` uses a moving average of a forcing term;
        ``"gas"`` uses the likelihood score. See the module docstring.
    forcing : {"auto", "normal-product", "abs-difference"}
        Patton's forcing term. ``"auto"`` picks the normal product for
        elliptical families and the absolute difference otherwise, which is what
        Patton did. Ignored by the GAS driver, which needs no such choice.
    lags : int
        Length of the moving-average window. Patton used 10.
    index : int
        Which parameter varies, when the family has more than one. Defaults to
        the first, which is the correlation for a t copula.
    bounds : (float, float), optional
        Override the working range of the varying parameter.

    Notes
    -----
    The likelihood is evaluated observation by observation, because the
    parameter differs at every one. Expect roughly 50 microseconds per
    observation per likelihood evaluation -- a few seconds to fit a decade of
    daily data, not milliseconds.

    Examples
    --------
    >>> import rcopula as rc
    >>> from rcopula.dynamic import DynamicCopula
    >>> model = DynamicCopula(rc.ClaytonCopula(1.0), coefficients=(0.1, -0.5, 0.9))
    >>> model.driver
    'patton'
    >>> model.forcing
    'abs-difference'
    """

    def __init__(
        self,
        family: Copula,
        *,
        coefficients: ArrayLike,
        driver: Driver = "patton",
        forcing: Forcing = "auto",
        lags: int = 10,
        index: int = 0,
        bounds: tuple[float, float] | None = None,
    ) -> None:
        if family.dim != 2:
            raise ValueError(
                f"a dynamic copula is bivariate; got dim={family.dim}. For a "
                "time-varying correlation matrix in higher dimensions use fit_dcc."
            )
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.shape != (3,):
            raise ValueError(
                f"coefficients must be (omega, alpha, beta); got shape {coefficients.shape}"
            )
        if driver not in ("patton", "gas"):
            raise ValueError(f"driver must be 'patton' or 'gas', got {driver!r}")

        self.family = family
        self.coefficients = coefficients
        self.driver = driver
        self.lags = int(lags)
        self.index = int(index)
        self.link = _default_link(family, self.index, bounds)

        if forcing == "auto":
            forcing = (
                "normal-product"
                if isinstance(family, GaussianCopula | StudentCopula)
                else "abs-difference"
            )
        self.forcing = forcing
        self._df = float(family.params[-1]) if isinstance(family, StudentCopula) else None

    # -- internals ---------------------------------------------------------

    def _params_at(self, theta: float) -> NDArray[np.float64]:
        """The full parameter vector with the varying entry replaced."""
        params = np.array(self.family.params, dtype=float)
        params[self.index] = theta
        return params

    def _prepare(self, u: NDArray[np.float64]) -> NDArray[np.float64] | None:
        """Everything about the data the log-density needs, computed once."""
        entry = _FAST_LOGPDF.get(type(self.family).__name__)
        if entry is None:
            return None
        return np.asarray(entry[0](u, self._df), dtype=float)

    def _logpdf_rows(
        self,
        u: NDArray[np.float64],
        thetas: NDArray[np.float64],
        prepared: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """``log c(u_t; theta_t)`` with a different parameter on every row.

        Uses the vectorised expression for this family when there is one and the
        whole path lies where it is valid; otherwise evaluates the family itself
        once per row, which is always available and always correct.
        """
        entry = _FAST_LOGPDF.get(type(self.family).__name__)
        if entry is not None:
            data = self._prepare(u) if prepared is None else prepared
            value = entry[1](data, thetas, self._df)
            if value is not None:
                return np.asarray(value, dtype=float)
        out = np.empty(u.shape[0], dtype=float)
        for t in range(u.shape[0]):
            out[t] = self.family._logpdf(u[t : t + 1], self._params_at(thetas[t]))[0]
        return out

    def _logpdf_at(
        self,
        row: NDArray[np.float64],
        theta: float,
        prepared: NDArray[np.float64] | None = None,
    ) -> float:
        return float(self._logpdf_rows(row.reshape(1, 2), np.array([theta]), prepared)[0])

    def _score(
        self,
        row: NDArray[np.float64],
        linked: float,
        prepared: NDArray[np.float64] | None = None,
    ) -> float:
        """d log c / d f by central difference on the linked scale.

        Doing it on the linked scale rather than the natural one is what makes
        the GAS update respect the parameter's domain automatically: the chain
        rule through the link is included, and it vanishes at the boundary.
        """
        up = self._logpdf_at(row, float(self.link(linked + _SCORE_STEP)), prepared)
        down = self._logpdf_at(row, float(self.link(linked - _SCORE_STEP)), prepared)
        if not (np.isfinite(up) and np.isfinite(down)):
            return 0.0
        return (up - down) / (2.0 * _SCORE_STEP)

    # -- public ------------------------------------------------------------

    def filter(self, u: ArrayLike) -> DynamicPath:
        """Run the recursion over ``u`` and return the parameter path.

        Parameters
        ----------
        u : array_like, shape (n, 2)
            Pseudo-observations in :math:`(0,1)^2`.

        Returns
        -------
        DynamicPath

        Examples
        --------
        >>> import numpy as np, rcopula as rc
        >>> from rcopula.dynamic import DynamicCopula
        >>> model = DynamicCopula(rc.GaussianCopula(0.5), coefficients=(0.0, 0.0, 0.0))
        >>> u = rc.GaussianCopula(0.5).rvs(50, random_state=0)
        >>> path = model.filter(u).path
        >>> bool(np.allclose(path, 0.0))     # omega = alpha = beta = 0 pins it at Lambda(0)
        True
        """
        u = np.asarray(u, dtype=float)
        if u.ndim != 2 or u.shape[1] != 2:
            raise ValueError(f"u must have shape (n, 2), got {u.shape}")
        n = u.shape[0]
        omega, alpha, beta = self.coefficients

        theta = np.empty(n, dtype=float)
        linked = np.empty(n, dtype=float)
        contributions = np.empty(n, dtype=float)

        start = self.link.inverse(float(self.family.params[self.index]))

        if self.driver == "patton":
            averaged = _moving_average(_forcing_values(u, self.forcing, self._df), self.lags)
            previous = float(self.link(start))
            # The Patton recursion never looks at the density, so the whole path
            # can be built first and the log-density evaluated in one shot.
            for t in range(n):
                # Patton writes beta against the lagged parameter on its natural
                # scale, not the linked one. Kept as written.
                linked[t] = omega + beta * self.link.standardise(previous) + alpha * averaged[t]
                theta[t] = float(self.link(linked[t]))
                previous = theta[t]
            contributions = self._logpdf_rows(u, theta)
        else:
            prepared = self._prepare(u)
            current = start
            for t in range(n):
                linked[t] = current
                theta[t] = float(self.link(current))
                row = None if prepared is None else prepared[t : t + 1]
                contributions[t] = self._logpdf_at(u[t], theta[t], row)
                current = omega + beta * current + alpha * self._score(u[t], current, row)

        contributions = np.where(np.isfinite(contributions), contributions, -1e10)
        return DynamicPath(path=theta, linked=linked, loglik_contributions=contributions)

    def loglik(self, u: ArrayLike) -> float:
        """Log-likelihood of ``u`` under the recursion."""
        return self.filter(u).loglik

    def with_coefficients(self, coefficients: ArrayLike) -> DynamicCopula:
        """A copy with different ``(omega, alpha, beta)``."""
        return DynamicCopula(
            self.family,
            coefficients=coefficients,
            driver=self.driver,
            forcing=self.forcing,
            lags=self.lags,
            index=self.index,
            bounds=(self.link.lower, self.link.upper),
        )

    def simulate(
        self, size: int, *, random_state: Any = None, burn_in: int = 200
    ) -> NDArray[np.float64]:
        """Draw a path from the model.

        The recursion feeds on its own output, so this is genuinely sequential:
        each observation is drawn from the copula the previous ones implied.

        Parameters
        ----------
        size : int
            Number of observations to return.
        random_state : None, int or Generator
        burn_in : int
            Draws discarded first, so the returned path does not depend on where
            ``family.params`` happened to start.

        Examples
        --------
        >>> import numpy as np, rcopula as rc
        >>> from rcopula.dynamic import DynamicCopula
        >>> model = DynamicCopula(rc.GaussianCopula(0.0), coefficients=(0.05, 0.2, 0.9))
        >>> u = model.simulate(200, random_state=1)
        >>> u.shape
        (200, 2)
        """
        rng = np.random.default_rng(random_state)
        total = int(size) + int(burn_in)
        omega, alpha, beta = self.coefficients

        draws = np.empty((total, 2), dtype=float)
        recent: list[float] = []
        previous = float(self.family.params[self.index])
        current = self.link.inverse(previous)
        mean_forcing: float | None = None

        for t in range(total):
            if self.driver == "patton":
                if len(recent) < self.lags:
                    # No window yet. Using the running mean of whatever exists
                    # keeps the burn-in from starting at an arbitrary value.
                    average = float(np.mean(recent)) if recent else (mean_forcing or 0.0)
                else:
                    average = float(np.mean(recent[-self.lags :]))
                linked = omega + beta * self.link.standardise(previous) + alpha * average
            else:
                linked = current

            theta = float(self.link(linked))
            row = self.family.with_params(self._params_at(theta)).rvs(1, random_state=rng)
            draws[t] = row[0]

            if self.driver == "patton":
                value = _forcing_values(draws[t : t + 1], self.forcing, self._df)[0]
                recent.append(float(value))
                mean_forcing = float(np.mean(recent))
                previous = theta
            else:
                current = omega + beta * current + alpha * self._score(draws[t], current)

        return draws[burn_in:]

    def forecast(
        self,
        u: ArrayLike,
        horizon: int,
        *,
        draws: int = 2000,
        random_state: Any = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Simulate the parameter forward from the end of ``u``.

        The recursion is driven by future observations, so beyond one step ahead
        the parameter is a random variable rather than a number. This returns
        its distribution.

        Returns
        -------
        dict
            ``"mean"``, ``"median"``, ``"lower"`` and ``"upper"`` (5th and 95th
            percentiles), each of length ``horizon``.

        Examples
        --------
        >>> import rcopula as rc
        >>> from rcopula.dynamic import DynamicCopula
        >>> model = DynamicCopula(rc.GaussianCopula(0.4), coefficients=(0.05, 0.1, 0.85))
        >>> u = model.simulate(300, random_state=0)
        >>> ahead = model.forecast(u, horizon=5, draws=200, random_state=1)
        >>> sorted(ahead)
        ['lower', 'mean', 'median', 'upper']
        >>> ahead["mean"].shape
        (5,)
        """
        u = np.asarray(u, dtype=float)
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")
        rng = np.random.default_rng(random_state)
        filtered = self.filter(u)
        omega, alpha, beta = self.coefficients

        paths = np.empty((draws, horizon), dtype=float)
        history = _forcing_values(u, self.forcing, self._df) if self.driver == "patton" else None

        for b in range(draws):
            previous = float(filtered.path[-1])
            current = float(filtered.linked[-1])
            window = list(history[-self.lags :]) if history is not None else []
            for h in range(horizon):
                if self.driver == "patton":
                    linked = (
                        omega
                        + beta * self.link.standardise(previous)
                        + alpha * float(np.mean(window))
                    )
                else:
                    linked = omega + beta * current + alpha * 0.0  # score is mean zero
                theta = float(self.link(linked))
                paths[b, h] = theta
                row = self.family.with_params(self._params_at(theta)).rvs(1, random_state=rng)
                if self.driver == "patton":
                    window.append(float(_forcing_values(row, self.forcing, self._df)[0]))
                    window = window[-self.lags :]
                    previous = theta
                else:
                    current = omega + beta * current + alpha * self._score(row[0], current)

        return {
            "mean": np.asarray(paths.mean(axis=0)),
            "median": np.asarray(np.median(paths, axis=0)),
            "lower": np.asarray(np.percentile(paths, 5, axis=0)),
            "upper": np.asarray(np.percentile(paths, 95, axis=0)),
        }

    def __repr__(self) -> str:
        omega, alpha, beta = self.coefficients
        return (
            f"DynamicCopula({self.family.describe()}, driver={self.driver!r}, "
            f"omega={omega:.4g}, alpha={alpha:.4g}, beta={beta:.4g})"
        )


# --------------------------------------------------------------------------
# estimation
# --------------------------------------------------------------------------


@dataclass
class DynamicFitResult:
    """A fitted time-varying copula.

    Attributes
    ----------
    model : DynamicCopula
        The model at the estimated coefficients.
    coefficients : ndarray
        ``(omega, alpha, beta)``.
    loglik : float
    path : ndarray
        The filtered parameter path.
    constant_loglik : float
        Log-likelihood of the same family with a constant parameter, fitted by
        maximum likelihood. The comparison is the point of the exercise.
    n_obs : int
    converged : bool
    """

    model: DynamicCopula
    coefficients: NDArray[np.float64]
    loglik: float
    path: NDArray[np.float64]
    constant_loglik: float
    constant_param: float
    n_obs: int
    converged: bool
    message: str = ""
    _u: NDArray[np.float64] = field(repr=False, default_factory=lambda: np.empty((0, 2)))

    @property
    def n_params(self) -> int:
        """Three, always: omega, alpha and beta."""
        return 3

    @property
    def aic(self) -> float:
        """Akaike information criterion."""
        return float(2 * self.n_params - 2 * self.loglik)

    @property
    def bic(self) -> float:
        """Bayesian information criterion."""
        return float(self.n_params * np.log(self.n_obs) - 2 * self.loglik)

    @property
    def persistence(self) -> float:
        """The autoregressive coefficient beta.

        Near 1 the parameter drifts slowly and the recursion is close to a unit
        root; near 0 it is essentially the constant model with noise.
        """
        return float(self.coefficients[2])

    def constancy_test(self) -> tuple[float, float]:
        """Likelihood ratio against a constant copula.

        Returns
        -------
        statistic, pvalue

        Notes
        -----
        The null sits on the boundary of the parameter space in ``beta`` (a
        constant copula is ``alpha = 0`` with ``beta`` unidentified), so the
        chi-squared reference distribution is **not** exact -- this is the
        Davies problem, and the reported p-value is conservative-to-wrong in the
        usual way for a nuisance parameter identified only under the
        alternative. Treat it as a diagnostic, not a test. A bootstrap under the
        constant null is the honest version and is left to the caller.

        Examples
        --------
        >>> import rcopula as rc
        >>> from rcopula.dynamic import DynamicCopula, fit_dynamic
        >>> model = DynamicCopula(rc.GaussianCopula(0.0), coefficients=(0.1, 0.3, 0.8))
        >>> u = model.simulate(600, random_state=0)
        >>> res = fit_dynamic(u, rc.GaussianCopula(0.0))
        >>> statistic, pvalue = res.constancy_test()
        >>> bool(statistic > 0)
        True
        """
        statistic = float(2.0 * (self.loglik - self.constant_loglik))
        statistic = max(statistic, 0.0)
        return statistic, float(stats.chi2(2).sf(statistic))

    def summary(self) -> str:
        """A printable report.

        Examples
        --------
        >>> import rcopula as rc
        >>> from rcopula.dynamic import DynamicCopula, fit_dynamic
        >>> model = DynamicCopula(rc.GaussianCopula(0.0), coefficients=(0.1, 0.3, 0.8))
        >>> u = model.simulate(400, random_state=2)
        >>> text = fit_dynamic(u, rc.GaussianCopula(0.0)).summary()
        >>> print(text.splitlines()[0])
        Time-varying Gaussian copula, dim 2, rho=0.151034 (patton recursion)
        >>> "LR vs constant" in text
        True
        """
        omega, alpha, beta = self.coefficients
        statistic, pvalue = self.constancy_test()
        lines = [
            f"Time-varying {self.model.family.describe()} ({self.model.driver} recursion)",
            "=" * 68,
            f"  observations         {self.n_obs}",
            f"  omega                {omega: .6f}",
            f"  alpha                {alpha: .6f}",
            f"  beta (persistence)   {beta: .6f}",
            "",
            f"  log-likelihood       {self.loglik: .4f}",
            f"  AIC / BIC            {self.aic: .4f} / {self.bic:.4f}",
            "",
            f"  constant parameter   {self.constant_param: .6f}",
            f"  constant log-lik     {self.constant_loglik: .4f}",
            f"  LR vs constant       {statistic: .4f}  (nominal p = {pvalue:.4f})",
            "",
            f"  parameter range      [{self.path.min():.4f}, {self.path.max():.4f}]",
            f"  mean parameter       {self.path.mean(): .6f}",
        ]
        if not self.converged:
            lines += ["", f"  WARNING: optimiser did not converge -- {self.message}"]
        return "\n".join(lines)


def fit_dynamic(
    u: ArrayLike,
    family: Copula,
    *,
    driver: Driver = "patton",
    forcing: Forcing = "auto",
    lags: int = 10,
    index: int = 0,
    bounds: tuple[float, float] | None = None,
    start: ArrayLike | None = None,
) -> DynamicFitResult:
    """Fit a time-varying copula by maximum likelihood.

    Parameters
    ----------
    u : array_like, shape (n, 2)
        Pseudo-observations. Anything outside :math:`(0,1)` is converted with
        :func:`~rcopula.pseudo_obs` first, so raw innovations are accepted --
        but see the warning in the module docstring about *which* series to
        pass.
    family : Copula
        The bivariate family whose parameter varies.
    driver, forcing, lags, index, bounds
        As for :class:`DynamicCopula`.
    start : sequence of 3 floats, optional
        Starting ``(omega, alpha, beta)``. The default starts from a persistent
        recursion centred on the constant maximum-likelihood estimate, which is
        the standard choice and matters: this likelihood is not concave.

    Returns
    -------
    DynamicFitResult

    Examples
    --------
    >>> import rcopula as rc
    >>> from rcopula.dynamic import DynamicCopula, fit_dynamic
    >>> truth = DynamicCopula(rc.GaussianCopula(0.0), coefficients=(0.1, 0.4, 0.8))
    >>> u = truth.simulate(800, random_state=0)
    >>> res = fit_dynamic(u, rc.GaussianCopula(0.0))
    >>> bool(res.loglik > res.constant_loglik)
    True
    """
    u = np.asarray(u, dtype=float)
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError(f"u must have shape (n, 2), got {u.shape}")
    if u.min() <= 0.0 or u.max() >= 1.0:
        u = pseudo_obs(u)
    n = u.shape[0]

    from rcopula.fit import fit as fit_constant

    constant = fit_constant(family, u, method="mpl")
    constant_param = float(np.asarray(constant.params)[index])
    constant_loglik = float(constant.loglik)

    template = DynamicCopula(
        family.with_params(constant.params),
        coefficients=(0.0, 0.0, 0.0),
        driver=driver,
        forcing=forcing,
        lags=lags,
        index=index,
        bounds=bounds,
    )

    def objective(theta: NDArray[np.float64]) -> float:
        # A |beta| at or past 1 is a non-stationary recursion; the likelihood
        # there is not comparable to the rest and the optimiser will happily
        # wander into it if the data are close to independent.
        if not np.all(np.isfinite(theta)) or abs(theta[2]) >= 0.9999:
            return 1e12
        value = template.with_coefficients(theta).loglik(u)
        return float(-value) if np.isfinite(value) else 1e12

    if start is None:
        beta0 = 0.9
        if driver == "patton":
            # Solve the stationary point for omega so the recursion starts at
            # the constant estimate: link_inv(theta*) = omega + beta*theta* + alpha*mbar.
            averaged = _moving_average(_forcing_values(u, template.forcing, template._df), lags)
            mean_forcing = float(np.mean(averaged))
            alpha0 = 0.05 if template.forcing == "normal-product" else -0.05
            omega0 = (
                template.link.inverse(constant_param)
                - beta0 * template.link.standardise(constant_param)
                - alpha0 * mean_forcing
            )
        else:
            alpha0 = 0.02
            omega0 = (1.0 - beta0) * template.link.inverse(constant_param)
        starts = [
            np.array([omega0, alpha0, beta0]),
            np.array([omega0, -alpha0, beta0]),
            np.array([template.link.inverse(constant_param) * 0.1, alpha0, 0.5]),
        ]
    else:
        starts = [np.asarray(start, dtype=float)]

    best: optimize.OptimizeResult | None = None
    for guess in starts:
        # Nelder-Mead: the score-driven objective is only piecewise smooth once
        # the finite-difference score is involved, and a gradient method walks
        # straight into the flat region where beta saturates.
        candidate = optimize.minimize(
            objective,
            guess,
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-8},
        )
        if best is None or candidate.fun < best.fun:
            best = candidate
    assert best is not None

    fitted = template.with_coefficients(best.x)
    path = fitted.filter(u)
    return DynamicFitResult(
        model=fitted,
        coefficients=np.asarray(best.x, dtype=float),
        loglik=path.loglik,
        path=path.path,
        constant_loglik=constant_loglik,
        constant_param=constant_param,
        n_obs=n,
        converged=bool(best.success),
        message=str(best.message),
        _u=u,
    )


# --------------------------------------------------------------------------
# DCC
# --------------------------------------------------------------------------


@dataclass
class DccResult:
    """A filtered dynamic conditional correlation model.

    Attributes
    ----------
    correlations : ndarray, shape (n, d, d)
        The correlation matrix at each observation.
    a, b : float
        The DCC coefficients. ``a`` is the news impact, ``b`` the persistence.
    loglik : float
        Copula log-likelihood, i.e. of the Gaussian (or t) copula density at the
        filtered correlations -- *not* the joint likelihood of the data.
    unconditional : ndarray, shape (d, d)
        The target correlation matrix the recursion reverts to.
    df : float or None
        Degrees of freedom, if a t copula was used.
    """

    correlations: NDArray[np.float64]
    a: float
    b: float
    loglik: float
    unconditional: NDArray[np.float64]
    df: float | None
    converged: bool
    n_obs: int

    @property
    def persistence(self) -> float:
        """``a + b``. At 1 the recursion has a unit root and never reverts."""
        return float(self.a + self.b)

    def pair(self, i: int, j: int) -> NDArray[np.float64]:
        """The correlation between coordinates ``i`` and ``j`` over time."""
        return np.asarray(self.correlations[:, i, j])

    def copulas(self) -> list[Copula]:
        """One copula per observation, for downstream use.

        Building ``n`` copula objects is not free; do it when the models are
        wanted individually, not to compute a likelihood.
        """
        from rcopula.core.elliptical import P2p

        dim = self.correlations.shape[1]
        out: list[Copula] = []
        for matrix in self.correlations:
            # An unstructured elliptical copula is parameterised by the flat
            # upper triangle, not by the matrix.
            flat = P2p(matrix)
            if self.df is None:
                out.append(GaussianCopula(flat, dim=dim, dispstr="un"))
            else:
                out.append(StudentCopula(flat, df=self.df, dim=dim, dispstr="un"))
        return out

    def summary(self) -> str:
        """A printable report."""
        return "\n".join(
            [
                f"Dynamic conditional correlation, d = {self.correlations.shape[1]}"
                + ("" if self.df is None else f", t copula with df = {self.df:.3f}"),
                "=" * 68,
                f"  observations         {self.n_obs}",
                f"  a (news impact)      {self.a: .6f}",
                f"  b (persistence)      {self.b: .6f}",
                f"  a + b                {self.persistence: .6f}",
                f"  copula log-lik       {self.loglik: .4f}",
                "",
                "  mean correlation     "
                + f"{np.mean(self.correlations[:, 0, 1]): .6f} (first pair)",
                "  range                "
                + f"[{self.correlations[:, 0, 1].min():.4f}, "
                + f"{self.correlations[:, 0, 1].max():.4f}]",
            ]
        )


def _dcc_filter(
    z: NDArray[np.float64], target: NDArray[np.float64], a: float, b: float
) -> NDArray[np.float64]:
    """Engle's recursion, returning correlation matrices.

    ``Q_t = (1 - a - b) S + a z_{t-1} z_{t-1}' + b Q_{t-1}``, then rescaled to
    unit diagonal. The rescaling is what makes it a correlation and is also why
    the two-step estimator is inconsistent (Aielli 2013): ``S`` is the target of
    ``Q``, not of ``R``. The bias is small at the persistences seen in practice
    and this implementation does not correct it.
    """
    n, d = z.shape
    correlations = np.empty((n, d, d), dtype=float)
    q = target.copy()
    constant = (1.0 - a - b) * target
    for t in range(n):
        if t > 0:
            outer = np.outer(z[t - 1], z[t - 1])
            q = constant + a * outer + b * q
        scale = 1.0 / np.sqrt(np.clip(np.diag(q), 1e-12, None))
        correlations[t] = q * scale[:, None] * scale[None, :]
        np.fill_diagonal(correlations[t], 1.0)
    return correlations


def _dcc_loglik(
    z: NDArray[np.float64], correlations: NDArray[np.float64], df: float | None
) -> float:
    """Copula log-likelihood at a sequence of correlation matrices.

    Written directly rather than through ``GaussianCopula.logpdf`` because that
    would rebuild and re-factorise a copula object at every observation.
    """
    d = z.shape[1]
    total = 0.0
    for t in range(z.shape[0]):
        matrix = correlations[t]
        try:
            factor = linalg.cholesky(matrix, lower=True)
        except linalg.LinAlgError:
            return -np.inf
        log_det = 2.0 * float(np.sum(np.log(np.diag(factor))))
        solved = linalg.solve_triangular(factor, z[t], lower=True)
        quadratic = float(solved @ solved)
        if df is None:
            total += -0.5 * log_det - 0.5 * (quadratic - float(z[t] @ z[t]))
        else:
            # log c = log f_joint(z) - sum_j log f_margin(z_j), both Student-t
            # with the same df. The margins are what make it a copula density
            # rather than a multivariate density.
            log_joint = (
                float(np.log(np.pi * df) * (-d / 2.0))
                + float(special.gammaln((df + d) / 2.0) - special.gammaln(df / 2.0))
                - 0.5 * log_det
                - (df + d) / 2.0 * np.log1p(quadratic / df)
            )
            log_margins = float(np.sum(stats.t(df).logpdf(z[t])))
            total += log_joint - log_margins
    return total


def fit_dcc(
    u: ArrayLike,
    *,
    df: float | None = None,
    start: tuple[float, float] = (0.03, 0.95),
) -> DccResult:
    r"""Fit Engle's DCC to the copula of ``u``.

    The multivariate answer to :class:`DynamicCopula`: instead of one parameter
    moving, the whole correlation matrix does, driven by the outer product of
    the last observation.

    Parameters
    ----------
    u : array_like, shape (n, d)
        Pseudo-observations. Values outside :math:`(0,1)` trigger a conversion.
    df : float, optional
        Fit a t copula with this many degrees of freedom instead of a Gaussian
        one. Not estimated -- profile it by calling this over a grid, which is
        what practitioners do anyway because the df likelihood is flat.
    start : (float, float)
        Starting ``(a, b)``.

    Returns
    -------
    DccResult

    Notes
    -----
    ``a`` and ``b`` are constrained to be non-negative with ``a + b < 1``, which
    is sufficient (not necessary) for :math:`Q_t` to stay positive definite.

    Examples
    --------
    >>> import numpy as np, rcopula as rc
    >>> from rcopula.dynamic import fit_dcc
    >>> rng = np.random.default_rng(0)
    >>> u = rc.GaussianCopula(0.5, dim=3, dispstr="ex").rvs(400, random_state=0)
    >>> res = fit_dcc(u)
    >>> res.correlations.shape
    (400, 3, 3)
    >>> bool(0.0 <= res.persistence < 1.0)
    True
    """
    u = np.asarray(u, dtype=float)
    if u.ndim != 2 or u.shape[1] < 2:
        raise ValueError(f"u must have shape (n, d) with d >= 2, got {u.shape}")
    if u.min() <= 0.0 or u.max() >= 1.0:
        u = pseudo_obs(u)

    clipped = np.clip(u, _QUANTILE_CLIP, 1 - _QUANTILE_CLIP)
    z = stats.norm.ppf(clipped) if df is None else stats.t(df).ppf(clipped)
    z = np.asarray(z, dtype=float)

    # The target is the unconditional correlation of the transformed data. Using
    # correlation targeting rather than estimating S jointly is Engle's own
    # two-step device and cuts the parameter count from d(d-1)/2 + 2 to 2.
    target = np.corrcoef(z, rowvar=False)

    def objective(theta: NDArray[np.float64]) -> float:
        a, b = float(theta[0]), float(theta[1])
        if a < 0 or b < 0 or a + b >= 0.9999:
            return 1e12
        value = _dcc_loglik(z, _dcc_filter(z, target, a, b), df)
        return float(-value) if np.isfinite(value) else 1e12

    result = optimize.minimize(
        objective,
        np.asarray(start, dtype=float),
        method="Nelder-Mead",
        options={"maxiter": 400, "xatol": 1e-5, "fatol": 1e-6},
    )
    a, b = float(result.x[0]), float(result.x[1])
    correlations = _dcc_filter(z, target, a, b)
    return DccResult(
        correlations=correlations,
        a=a,
        b=b,
        loglik=float(_dcc_loglik(z, correlations, df)),
        unconditional=target,
        df=df,
        converged=bool(result.success),
        n_obs=int(z.shape[0]),
    )
