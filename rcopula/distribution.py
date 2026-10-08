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
    """The methods a single-variable distribution needs to be used as a margin.

    A margin describes one variable on its own (its scale and shape), for
    example ``scipy.stats.norm(loc=0, scale=1)``. You never instantiate
    ``Margin``; it only documents what :class:`CopulaDistribution` expects.

    Satisfied by every ``scipy.stats`` frozen distribution, and by anything else
    exposing the same three methods:

    - ``cdf(x)``: cumulative probability ``P(X <= x)`` for array_like ``x``.
    - ``pdf(x)``: density at ``x``.
    - ``ppf(q)``: quantile function, the inverse of ``cdf``, for ``q`` in
      ``[0, 1]``.

    :class:`CopulaDistribution` also accepts discrete margins that provide
    ``pmf`` instead of ``pdf`` (such as scipy's frozen discrete distributions).

    Examples
    --------
    >>> from scipy import stats
    >>> from rcopula.distribution import Margin
    >>> isinstance(stats.norm(), Margin)
    True
    """

    def cdf(self, x: ArrayLike) -> Any: ...
    def pdf(self, x: ArrayLike) -> Any: ...
    def ppf(self, q: ArrayLike) -> Any: ...


class CopulaDistribution:
    """A joint distribution made from a copula plus one distribution per variable.

    Use this when you want realistic joint samples or joint probabilities on
    the original scale of your data: choose each variable's own distribution
    (the *margins*, e.g. a normal for one, an exponential for another) and a
    copula for how they move together. The result has those exact margins and
    that exact dependence (Sklar's theorem).

    Parameters
    ----------
    copula : Copula
        The dependence structure, e.g. ``ClaytonCopula(2.0, dim=2)``. Must be
        fully specified (no ``nan`` parameters) before evaluating or sampling.
    margins : Margin or list of Margin
        One distribution per variable, in column order, typically scipy frozen
        distributions such as ``stats.norm(loc=1, scale=2)``. A list or tuple
        must have exactly ``copula.dim`` entries; a single distribution is used
        for every variable. Each margin needs ``cdf``, ``ppf`` and either
        ``pdf`` (continuous) or ``pmf`` (discrete).
    names : list of str or None, default None
        Column names, length ``copula.dim``. When given, :meth:`rvs` returns a
        ``pandas.DataFrame`` with these columns instead of an array.

    Attributes
    ----------
    copula : Copula
        The copula passed in.
    margins : list of Margin
        The margins, one per variable (a single margin is repeated).
    discrete : numpy.ndarray of bool, shape (d,)
        ``True`` for variables whose margin is discrete (has ``pmf`` but no
        ``pdf``).
    names : list of str or None
        Column names, or ``None``.
    dim : int
        Number of variables.

    Raises
    ------
    TypeError
        If ``copula`` is not a :class:`~rcopula.core.base.Copula`, or a margin
        lacks the required methods.
    ValueError
        If the number of margins or names does not match ``copula.dim``.

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
        """Number of variables in the distribution (same as the copula's ``dim``).

        Returns
        -------
        int
            The dimension, at least 2.
        """
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
        r"""Probability that every variable is at or below the given values.

        For two variables and a point ``(a, b)`` this is
        ``P(X_1 <= a and X_2 <= b)``, the joint cumulative distribution
        function :math:`C(F_1(x_1), \dots, F_d(x_d))`.

        Parameters
        ----------
        x : array_like of float or pandas.DataFrame, shape (n, d) or (d,)
            Points on the original scale of the variables, one row per point.
            A 1-D input is treated as a single point. A DataFrame is used by
            column position, not by name.

        Returns
        -------
        numpy.ndarray of float, shape (n,)
            Probabilities in ``[0, 1]``.

        Raises
        ------
        ValueError
            If ``x`` does not have ``d`` columns, or the copula is unfitted.
        """
        return self.copula.cdf(self._to_uniform(self._validate_x(x)))

    def logpdf(self, x: ArrayLike) -> NDArray[np.float64]:
        r"""Natural logarithm of the joint density at each point.

        Use this rather than ``log(pdf(x))`` for likelihoods: it stays accurate
        when the density is extremely small.

        Parameters
        ----------
        x : array_like of float or pandas.DataFrame, shape (n, d) or (d,)
            Points on the original scale of the variables, one row per point.
            A 1-D input is treated as a single point.

        Returns
        -------
        numpy.ndarray of float, shape (n,)
            Log density (or log mass, with discrete margins); ``-inf`` where the
            density is zero.

        Raises
        ------
        ValueError
            If ``x`` does not have ``d`` columns, or the copula is unfitted.

        Notes
        -----
        By the chain rule,
        :math:`h(\mathbf{x}) = c(F_1(x_1),\dots)\prod_j f_j(x_j)`, so the log
        density is the copula log density plus the marginal log densities. Doing
        it in logs matters: in even moderate dimensions the product of marginal
        densities underflows long before the joint density is genuinely zero.

        With a discrete margin the quantity is a mass rather than a density and
        there is no log-domain form to stay in -- the inclusion-exclusion sum has
        to be taken in the linear domain before the log -- so that case falls
        back to ``log(pdf(x))`` and inherits its underflow behaviour.
        """
        arr = self._validate_x(x)
        if self.discrete.any():
            with np.errstate(divide="ignore"):
                return np.asarray(np.log(self.pdf(arr)))
        u = self._to_uniform(arr)
        with np.errstate(divide="ignore"):
            marginal = np.sum(
                [np.log(m.pdf(arr[:, j])) for j, m in enumerate(self.margins)], axis=0
            )
        return np.asarray(self.copula.logpdf(u) + marginal)

    def pdf(self, x: ArrayLike) -> NDArray[np.float64]:
        r"""Joint density at each point (or probability mass, if margins are discrete).

        Parameters
        ----------
        x : array_like of float or pandas.DataFrame, shape (n, d) or (d,)
            Points on the original scale of the variables, one row per point.
            A 1-D input is treated as a single point.

        Returns
        -------
        numpy.ndarray of float, shape (n,)
            Joint density for continuous margins, probability mass for all
            discrete margins, or the mixed density/mass otherwise.

        Raises
        ------
        ValueError
            If ``x`` does not have ``d`` columns, or the copula is unfitted.

        Notes
        -----
        Joint density, or mass, or the mixture of the two.

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
        """Generate random samples on the original scale of the variables.

        Parameters
        ----------
        size : int, default 1
            Number of observations (rows) to draw. Must be non-negative.
        random_state : int, numpy.random.Generator or None, default None
            Seed or generator for reproducible draws; ``None`` gives fresh
            randomness.

        Returns
        -------
        numpy.ndarray of float, shape (size, d), or pandas.DataFrame
            Samples, one row per observation. A ``DataFrame`` with columns
            :attr:`names` is returned when ``names`` was given.

        Raises
        ------
        ValueError
            If the copula is unfitted or ``size < 0``.

        Notes
        -----
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
        """Convert data to the unit scale by applying each margin's CDF to its column.

        The result is what the copula itself sees: column ``j`` becomes
        ``F_j(x[:, j])``, a value in ``[0, 1]`` -- the copula's own arguments.

        Parameters
        ----------
        x : array_like of float or pandas.DataFrame, shape (n, d) or (d,)
            Points on the original scale, one row per point.

        Returns
        -------
        numpy.ndarray of float, shape (n, d)
            Marginal probabilities in ``[0, 1]``.

        Raises
        ------
        ValueError
            If ``x`` does not have ``d`` columns.
        """
        return self._to_uniform(self._validate_x(x))

    def describe(self) -> str:
        """Short human-readable summary of the copula and the margins.

        Returns
        -------
        str
            One line, e.g.
            ``'Clayton copula, dim 2, theta=2 with margins [norm, expon]'``.
        """
        # `dist.name` is a scipy detail, not part of the Margin protocol, so
        # fall back to the class name for user-supplied margins.
        names = ", ".join(
            getattr(getattr(m, "dist", None), "name", type(m).__name__) for m in self.margins
        )
        return f"{self.copula.describe()} with margins [{names}]"

    def __repr__(self) -> str:
        return f"<CopulaDistribution: {self.describe()}>"
