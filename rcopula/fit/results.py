r"""The object returned by :func:`rcopula.fit`.

Follows the ``statsmodels`` Model/Results split rather than scikit-learn's
"return ``self``": copula inference needs standard errors, a covariance matrix,
a log-likelihood and a printable summary, and hanging those off a mutated
copula makes fitted and unfitted objects indistinguishable.

Notably, ``bse`` (standard errors) is the piece no other Python copula package
provides at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import stats

if TYPE_CHECKING:  # pragma: no cover
    from rcopula.core.base import Copula

__all__ = ["CopulaFitResult"]


class CopulaFitResult:
    """Result of fitting a copula.

    Attributes
    ----------
    copula : Copula
        The fitted copula, with estimated parameters in place.
    params : ndarray
        Estimated free parameters.
    param_names : tuple of str
        Names matching ``params``.
    cov_params : ndarray or None
        Asymptotic covariance matrix, or ``None`` when it was not computed.
    loglik : float
        Maximised log-likelihood (pseudo-likelihood for ``"mpl"``).
    n_obs : int
        Number of observations.
    method : str
        Estimation method used.
    converged : bool
        Whether the optimiser reported success.
    message : str
        Optimiser message, or a note for closed-form estimators.
    """

    def __init__(
        self,
        copula: Copula,
        params: NDArray[np.float64],
        param_names: tuple[str, ...],
        loglik: float,
        n_obs: int,
        method: str,
        cov_params: NDArray[np.float64] | None = None,
        converged: bool = True,
        message: str = "",
    ) -> None:
        self.copula = copula
        self.params = np.atleast_1d(np.asarray(params, dtype=np.float64))
        self.param_names = tuple(param_names)
        self.cov_params = cov_params
        self.loglik = float(loglik)
        self.n_obs = int(n_obs)
        self.method = method
        self.converged = bool(converged)
        self.message = message

    # ------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        """Number of estimated parameters."""
        return int(self.params.size)

    @property
    def bse(self) -> NDArray[np.float64] | None:
        """Asymptotic standard errors, or ``None`` if unavailable.

        This is what no other Python copula package offers.
        """
        if self.cov_params is None:
            return None
        return np.sqrt(np.diag(self.cov_params))

    @property
    def tvalues(self) -> NDArray[np.float64] | None:
        """Estimate divided by standard error."""
        se = self.bse
        return None if se is None else self.params / se

    @property
    def pvalues(self) -> NDArray[np.float64] | None:
        """Two-sided p-values against a zero parameter.

        Interpret with care: for most families zero is not a meaningful null
        (Gumbel's parameter space starts at 1), so these are informative only
        where zero really means independence.
        """
        t = self.tvalues
        return None if t is None else 2.0 * stats.norm.sf(np.abs(t))

    @property
    def aic(self) -> float:
        """Akaike information criterion, ``-2 loglik + 2 k``."""
        return -2.0 * self.loglik + 2.0 * self.n_params

    @property
    def bic(self) -> float:
        """Bayesian information criterion, ``-2 loglik + k log n``."""
        return -2.0 * self.loglik + self.n_params * np.log(self.n_obs)

    def conf_int(self, alpha: float = 0.05) -> NDArray[np.float64] | None:
        """Wald confidence intervals at level ``1 - alpha``.

        Returns an ``(n_params, 2)`` array, or ``None`` without a covariance
        matrix.
        """
        se = self.bse
        if se is None:
            return None
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        return np.column_stack([self.params - z * se, self.params + z * se])

    # ------------------------------------------------------------------

    def summary(self) -> str:
        """A printable summary table, in the spirit of R's ``summary.fitCopula``."""
        se = self.bse

        lines = [
            f"{self.copula.name} copula fit  (dim {self.copula.dim})",
            f"  method       : {self.method}",
            f"  observations : {self.n_obs}",
            f"  log-likelihood: {self.loglik:.6g}",
            f"  AIC / BIC    : {self.aic:.6g} / {self.bic:.6g}",
            "",
        ]

        if se is None:
            lines.append(f"{'parameter':<14}{'estimate':>14}")
            lines.append("-" * 28)
            for name, value in zip(self.param_names, self.params, strict=True):
                lines.append(f"{name:<14}{value:>14.6g}")
            lines.append("")
            lines.append("Standard errors were not computed for this fit.")
        else:
            t = self.params / se
            p = 2.0 * stats.norm.sf(np.abs(t))
            lines.append(f"{'parameter':<14}{'estimate':>12}{'std.err':>12}{'z':>10}{'P>|z|':>10}")
            lines.append("-" * 58)
            for i, name in enumerate(self.param_names):
                lines.append(
                    f"{name:<14}{self.params[i]:>12.6g}{se[i]:>12.6g}{t[i]:>10.3f}{p[i]:>10.4f}"
                )

        if not self.converged:
            lines += ["", f"WARNING: optimiser did not converge -- {self.message}"]
        return "\n".join(lines)

    def __repr__(self) -> str:
        shown = ", ".join(
            f"{n}={v:.6g}" for n, v in zip(self.param_names, self.params, strict=True)
        )
        return f"<CopulaFitResult {self.copula.name}({shown}) method={self.method!r}>"
