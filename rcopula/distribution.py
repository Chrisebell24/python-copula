r"""Multivariate distributions built from a copula and margins.

This is the second direction of Sklar's theorem. The first direction says every
joint distribution *decomposes* into margins and a copula; the second says any
copula and any margins can be *combined* into a valid joint distribution:

.. math::

    H(x_1, \dots, x_d) = C\bigl(F_1(x_1), \dots, F_d(x_d)\bigr).

That is the whole practical appeal. Fit each margin however suits it -- a fitted
Gamma for claim sizes, a Student-t for returns, a kernel estimate for something
awkward -- and choose the dependence structure separately.

R calls this object ``mvdc`` and identifies margins by name strings
(``"norm"``, ``"exp"``, ...) with parameters in a list of lists. Here margins are
**scipy frozen distributions**, which is both more flexible (anything with
``cdf``/``pdf``/``ppf`` works, including user-defined distributions) and far
harder to get wrong.

References
----------
Sklar, A. (1959). Fonctions de repartition a n dimensions et leurs marges.
    *Publications de l'Institut de Statistique de l'Universite de Paris* 8,
    229-231.
Joe, H. (2014). *Dependence Modeling with Copulas*. Chapman & Hall/CRC.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from rcopula.core.base import Copula

__all__ = ["CopulaDistribution", "Margin"]


@runtime_checkable
class Margin(Protocol):
    """What a marginal distribution must provide.

    Satisfied by every ``scipy.stats`` frozen distribution, and by anything else
    exposing the same three methods.
    """

    def cdf(self, x: ArrayLike) -> Any: ...
    def pdf(self, x: ArrayLike) -> Any: ...
    def ppf(self, q: ArrayLike) -> Any: ...


class CopulaDistribution:
    """A joint distribution assembled from a copula and its margins.

    Parameters
    ----------
    copula : Copula
        The dependence structure.
    margins : sequence of frozen distributions
        One per dimension. A single distribution is broadcast to all dimensions.
    names : sequence of str, optional
        Column names, used when input or output is a ``pandas`` frame.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> from rcopula import ClaytonCopula, CopulaDistribution
    >>> mv = CopulaDistribution(
    ...     ClaytonCopula(2.0, dim=2),
    ...     margins=[stats.norm(loc=1, scale=2), stats.expon(scale=1 / 3)],
    ... )
    >>> x = mv.rvs(5000, random_state=0)
    >>> x.shape
    (5000, 2)

    The margins come out as specified:

    >>> bool(abs(x[:, 0].mean() - 1.0) < 0.1)
    True
    >>> bool(abs(x[:, 1].mean() - 1 / 3) < 0.02)
    True

    ...while the dependence is the copula's:

    >>> from scipy.stats import kendalltau
    >>> bool(abs(kendalltau(x[:, 0], x[:, 1]).statistic - 0.5) < 0.03)
    True

    Evaluation at a point:

    >>> float(round(mv.cdf([[1.0, 0.5]])[0], 10))
    0.4633938511
    >>> float(round(mv.pdf([[1.0, 0.5]])[0], 10))
    0.1460418727

    A single margin is broadcast:

    >>> CopulaDistribution(ClaytonCopula(2.0, dim=3), stats.norm()).dim
    3
    """

    def __init__(
        self,
        copula: Copula,
        margins: Margin | list[Margin],
        names: list[str] | None = None,
    ) -> None:
        if not isinstance(copula, Copula):
            raise TypeError(f"copula must be a Copula instance, got {type(copula).__name__}")

        marg = list(margins) if isinstance(margins, (list, tuple)) else [margins] * copula.dim
        if len(marg) != copula.dim:
            raise ValueError(f"got {len(marg)} margin(s) for a copula of dimension {copula.dim}")
        for j, m in enumerate(marg):
            # A discrete margin has pmf where a continuous one has pdf. Both are
            # accepted; which is which decides how the density is computed.
            missing = [a for a in ("cdf", "ppf") if not hasattr(m, a)]
            if not hasattr(m, "pdf") and not hasattr(m, "pmf"):
                missing.append("pdf or pmf")
            if missing:
                raise TypeError(
                    f"margin {j} ({type(m).__name__}) is missing {missing}; "
                    "a scipy frozen distribution such as stats.norm(0, 1) works"
                )

        self.copula = copula
        self.margins = marg
        #: Which coordinates are discrete. A margin with ``pmf`` and no ``pdf``
        #: is discrete; scipy's frozen discrete distributions are exactly that.
        self.discrete = np.array(
            [not hasattr(m, "pdf") and hasattr(m, "pmf") for m in marg], dtype=bool
        )
        self.names = list(names) if names is not None else None
        if self.names is not None and len(self.names) != copula.dim:
            raise ValueError(f"got {len(self.names)} names for dimension {copula.dim}")

    @property
    def dim(self) -> int:
        """Dimension of the distribution."""
        return self.copula.dim

    # ------------------------------------------------------------------

    def _validate_x(self, x: ArrayLike) -> NDArray[np.float64]:
        frame = x if isinstance(x, pd.DataFrame) else None
        arr = np.asarray(frame.to_numpy() if frame is not None else x, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.dim:
            raise ValueError(
                f"x has {arr.shape[1]} column(s) but the distribution has dim={self.dim}"
            )
        return arr

    def _to_uniform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.column_stack([m.cdf(x[:, j]) for j, m in enumerate(self.margins)])

    # ------------------------------------------------------------------

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        r"""Joint distribution function :math:`C(F_1(x_1), \dots, F_d(x_d))`."""
        return self.copula.cdf(self._to_uniform(self._validate_x(x)))

    def logpdf(self, x: ArrayLike) -> NDArray[np.float64]:
        r"""Log joint density.

        By the chain rule,
        :math:`h(\mathbf{x}) = c(F_1(x_1),\dots)\prod_j f_j(x_j)`, so the log
        density is the copula log density plus the marginal log densities. Doing
        it in logs matters: in even moderate dimensions the product of marginal
        densities underflows long before the joint density is genuinely zero.
        """
        with np.errstate(divide="ignore"):
            return np.log(self.pdf(x))

    def pdf(self, x: ArrayLike) -> NDArray[np.float64]:
        r"""Joint density, or mass, or the mixture of the two.

        With continuous margins this is
        :math:`c(F_1(x_1),\dots)\prod_j f_j(x_j)`. With any discrete margin it
        is not a derivative in that coordinate but a finite difference, so the
        work is handed to :func:`rcopula.discrete.mixed_pdf` -- see that module
        for what identifiability means once a margin has atoms.
        """
        arr = self._validate_x(x)
        if self.discrete.any():
            from rcopula.discrete import mixed_pdf

            return mixed_pdf(self.copula, arr, self.margins, self.discrete)
        u = self._to_uniform(arr)
        with np.errstate(divide="ignore"):
            marginal = np.sum(
                [np.log(m.pdf(arr[:, j])) for j, m in enumerate(self.margins)], axis=0
            )
        return np.asarray(np.exp(self.copula.logpdf(u) + marginal))

    def rvs(
        self,
        size: int = 1,
        random_state: np.random.Generator | int | None = None,
    ) -> NDArray[np.float64] | pd.DataFrame:
        """Draw from the joint distribution.

        Samples the copula, then pushes each coordinate through the
        corresponding marginal quantile function.
        """
        u = self.copula.rvs(size, random_state=random_state)
        x = np.column_stack([m.ppf(u[:, j]) for j, m in enumerate(self.margins)])
        if self.names is not None:
            return pd.DataFrame(x, columns=self.names)
        return x

    # ------------------------------------------------------------------

    def marginal_cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        """The marginal CDFs applied columnwise -- the copula's own arguments."""
        return self._to_uniform(self._validate_x(x))

    def describe(self) -> str:
        """One-line summary."""
        # `dist.name` is a scipy detail, not part of the Margin protocol, so
        # fall back to the class name for user-supplied margins.
        names = ", ".join(
            getattr(getattr(m, "dist", None), "name", type(m).__name__) for m in self.margins
        )
        return f"{self.copula.describe()} with margins [{names}]"

    def __repr__(self) -> str:
        return f"<CopulaDistribution: {self.describe()}>"
