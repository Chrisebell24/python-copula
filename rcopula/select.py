r"""Automatic copula family selection.

Choosing a family is the step everyone has to do and nobody enjoys. R makes you
loop by hand: construct each candidate, call ``fitCopula``, collect the
log-likelihoods, remember which families are bivariate-only, and handle the ones
that fail to converge. This module does that once, properly.

.. code-block:: python

    ranking = rc.select_copula(u)
    print(ranking)          # a ranked table
    best = ranking.best     # the fitted copula, ready to use

The comparison is more than a beauty contest, because the candidates differ in
what they can *express*. Gaussian and Frank have no tail dependence at all;
Clayton has it only below, Gumbel only above; Student-t has it in both tails and
buys that with one extra parameter. So the ranking is really a statement about
which asymmetries the data insists on -- and the table reports
:math:`\lambda_L`, :math:`\lambda_U` and :math:`\tau` next to each score so that
statement is visible rather than buried.

Two cautions, both stated in the table rather than hidden:

* **AIC and BIC only compare fit against complexity.** The best of a bad set is
  still bad. Pass ``gof=True`` (or ``gof="mult"`` for the fast version) to test
  the winner against the *data* rather than against the other candidates.
* **Selection is itself an estimate.** With a few hundred observations the top
  few families are usually within noise of each other; a difference in AIC of
  less than about 2 is not evidence. :func:`cross_validate` is the more honest
  comparison when you can afford it.

============================  ================================================
:func:`select_copula`         Fit every admissible family and rank them.
:func:`cross_validate`        k-fold cross-validated log-likelihood.
:class:`SelectionResult`      The ranked table, plus the winning copula.
:data:`FAMILIES`              The candidate registry, by name.
============================  ================================================

References
----------
Akaike, H. (1974). A new look at the statistical model identification.
    *IEEE Transactions on Automatic Control* 19(6), 716-723.
Schwarz, G. (1978). Estimating the dimension of a model.
    *Annals of Statistics* 6(2), 461-464.
Gronneberg, S. and Hjort, N. L. (2014). The copula information criteria.
    *Scandinavian Journal of Statistics* 41(2), 436-459.
    Why the naive AIC is biased for copulas fitted to pseudo-observations, and
    why cross-validation avoids the problem.
Genest, C., Remillard, B. and Beaudoin, D. (2009). Goodness-of-fit tests for
    copulas: a review and a power study.
    *Insurance: Mathematics and Economics* 44(2), 199-213.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from rcopula.core.archimedean import (
    AMHCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    JoeCopula,
)
from rcopula.core.base import Copula
from rcopula.core.elliptical import GaussianCopula, StudentCopula
from rcopula.core.extreme_value import (
    GalambosCopula,
    HuslerReissCopula,
    TawnCopula,
    TEVCopula,
)
from rcopula.core.other import FGMCopula, IndependenceCopula, PlackettCopula
from rcopula.dependence import pseudo_obs
from rcopula.fit import fit
from rcopula.fit.results import CopulaFitResult
from rcopula.gof import gof_test

__all__ = [
    "FAMILIES",
    "FamilySpec",
    "SelectionResult",
    "cross_validate",
    "select_copula",
]

#: Recognised ranking criteria. ``aic``/``bic``/``xv`` are the useful ones;
#: ``loglik`` ignores complexity entirely and always prefers the richest family.
CRITERIA = ("aic", "bic", "loglik", "xv")


@dataclass(frozen=True)
class FamilySpec:
    """One candidate family, with the dimensions it is valid in.

    Attributes
    ----------
    name : str
        Key in :data:`FAMILIES`.
    factory : callable
        ``dim -> Copula``, an unfitted instance.
    max_dim : int or None
        Largest supported dimension; ``None`` means any.
    groups : frozenset of str
        Group labels accepted by ``select_copula(families=...)``.
    """

    name: str
    factory: Callable[[int], Copula]
    max_dim: int | None
    groups: frozenset[str]

    def admissible(self, dim: int) -> bool:
        return self.max_dim is None or dim <= self.max_dim


def _spec(
    name: str,
    factory: Callable[[int], Copula],
    *groups: str,
    max_dim: int | None = None,
) -> FamilySpec:
    return FamilySpec(name, factory, max_dim, frozenset(groups) | {"all"})


#: The candidate registry. Keys are the names used in ``families=[...]`` and in
#: the result table; group labels (``"elliptical"``, ``"archimedean"``,
#: ``"extreme"``, ``"all"``) select several at once.
#:
#: Families excluded on purpose: Marshall-Olkin (its density is undefined on a
#: curve, so the likelihood is not comparable), the Frechet bounds (no density),
#: and FGM beyond ``d = 2`` (``2^d - d - 1`` parameters, 1013 at ``d = 10``).
FAMILIES: dict[str, FamilySpec] = {
    spec.name: spec
    for spec in (
        _spec("independence", lambda d: IndependenceCopula(d), "baseline"),
        _spec("gaussian", lambda d: GaussianCopula(dim=d), "elliptical"),
        _spec("student", lambda d: StudentCopula(dim=d), "elliptical"),
        _spec("clayton", lambda d: ClaytonCopula(dim=d), "archimedean"),
        _spec("gumbel", lambda d: GumbelCopula(dim=d), "archimedean", "extreme"),
        _spec("frank", lambda d: FrankCopula(dim=d), "archimedean"),
        _spec("joe", lambda d: JoeCopula(dim=d), "archimedean"),
        _spec("amh", lambda d: AMHCopula(dim=d), "archimedean"),
        _spec("galambos", lambda d: GalambosCopula(1.0), "extreme", max_dim=2),
        _spec("husler_reiss", lambda d: HuslerReissCopula(1.0), "extreme", max_dim=2),
        _spec("tawn", lambda d: TawnCopula(0.5), "extreme", max_dim=2),
        _spec("tev", lambda d: TEVCopula(0.5), "extreme", max_dim=2),
        _spec("plackett", lambda d: PlackettCopula(2.0), "other", max_dim=2),
        _spec("fgm", lambda d: FGMCopula(0.3), "other", max_dim=2),
    )
}


def _resolve(
    families: str | Sequence[str] | Sequence[Copula], dim: int
) -> list[tuple[str, Copula]]:
    """Turn the ``families`` argument into ``(name, unfitted copula)`` pairs."""
    if isinstance(families, str):
        chosen = [s for s in FAMILIES.values() if families in s.groups]
        if not chosen:
            raise ValueError(
                f"unknown family or group {families!r}; expected one of "
                f"{sorted(set(FAMILIES) | {g for s in FAMILIES.values() for g in s.groups})}"
            )
        usable = [s for s in chosen if s.admissible(dim)]
        if not usable:
            raise ValueError(f"no family in group {families!r} supports dim={dim}")
        return [(s.name, s.factory(dim)) for s in usable]

    items = list(families)
    if not items:
        raise ValueError("families is empty")

    if all(isinstance(item, Copula) for item in items):
        out = []
        seen: dict[str, int] = {}
        for item in items:
            assert isinstance(item, Copula)
            if item.dim != dim:
                raise ValueError(
                    f"{item.name} copula has dim={item.dim} but the data has {dim} columns"
                )
            key = item.name.lower()
            seen[key] = seen.get(key, 0) + 1
            out.append((key if seen[key] == 1 else f"{key}_{seen[key]}", item))
        return out

    resolved = []
    for item in items:
        if not isinstance(item, str):
            raise TypeError("families must be a group name, a list of names, or a list of copulas")
        if item not in FAMILIES:
            raise ValueError(f"unknown family {item!r}; expected one of {sorted(FAMILIES)}")
        spec = FAMILIES[item]
        if not spec.admissible(dim):
            raise ValueError(f"the {item} copula is limited to dim <= {spec.max_dim}, got {dim}")
        resolved.append((item, spec.factory(dim)))
    return resolved


def cross_validate(
    copula: Copula,
    data: ArrayLike,
    k: int = 10,
    method: str = "mpl",
    random_state: np.random.Generator | int | None = None,
    ties_method: str = "average",
) -> float:
    r"""k-fold cross-validated log-likelihood (R's ``xvCopula``).

    Fits on ``k-1`` folds and scores the held-out one, summed over folds and
    **multiplied by** :math:`n/(n - n/k)` so the result is on the scale of a
    full-sample log-likelihood and comparable across families -- the same
    convention R uses.

    This is the honest answer to a real problem: the ordinary AIC is biased for
    copulas, because the pseudo-observations are themselves estimated from the
    data and the usual "one penalty unit per parameter" accounting no longer
    holds (Gronneberg & Hjort 2014). Cross-validation sidesteps the bias by
    scoring on data the fit never saw. It costs ``k`` fits per family.

    Parameters
    ----------
    copula : Copula
        Family to score.
    data : array_like
        ``(n, d)`` observations; rank-transformed internally.
    k : int
        Number of folds.
    method : str
        Estimation method for each training fit.
    random_state : Generator, int or None
        Controls the fold assignment.

    Returns
    -------
    float
        Cross-validated log-likelihood; higher is better.

    Examples
    --------
    The generating family wins on data it generated:

    >>> import rcopula as rc
    >>> from rcopula.select import cross_validate
    >>> u = rc.ClaytonCopula(3.0).rvs(600, random_state=0)
    >>> right = cross_validate(rc.ClaytonCopula(), u, k=5, random_state=0)
    >>> wrong = cross_validate(rc.GumbelCopula(), u, k=5, random_state=0)
    >>> bool(right > wrong)
    True
    """
    u = pseudo_obs(data, ties_method=ties_method)
    n = u.shape[0]
    if not 2 <= k <= n:
        raise ValueError(f"k must be between 2 and n={n}, got {k}")

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    folds = np.array_split(rng.permutation(n), k)

    total = 0.0
    for held_out in folds:
        mask = np.ones(n, dtype=bool)
        mask[held_out] = False
        trained = fit(copula, u[mask], method=method, estimate_variance=False).copula
        # Re-rank each part on its own so the held-out scores are not computed
        # at pseudo-observations that depend on the training rows.
        total += float(np.sum(trained.logpdf(pseudo_obs(u[held_out]))))

    # Scale from "sum over folds of a (n/k)-sized score" to full-sample scale.
    return float(total * n / (n - n / k))


@dataclass(frozen=True)
class SelectionResult:
    """Outcome of :func:`select_copula`.

    Attributes
    ----------
    table : DataFrame
        One row per candidate, sorted best-first. Columns: ``n_params``,
        ``loglik``, ``aic``, ``bic``, ``tau``, ``lambda_lower``,
        ``lambda_upper``, ``converged``, ``message``, plus ``xv`` and the
        goodness-of-fit columns when those were requested.
    results : dict
        Family name to :class:`~rcopula.fit.results.CopulaFitResult`, including
        the ones that failed (absent if the fit raised).
    criterion : str
        Which column decided the ranking.
    """

    table: pd.DataFrame
    results: dict[str, CopulaFitResult]
    criterion: str

    @property
    def best_name(self) -> str:
        """Name of the winning family."""
        if self.table.empty:
            raise ValueError("no family could be fitted")
        return str(self.table.index[0])

    @property
    def best_result(self) -> CopulaFitResult:
        """The winning :class:`~rcopula.fit.results.CopulaFitResult`."""
        return self.results[self.best_name]

    @property
    def best(self) -> Copula:
        """The winning copula, fitted and ready to use."""
        return self.best_result.copula

    def summary(self) -> str:
        """A printable ranking table."""
        head = (
            f"Copula family selection  (n = {int(self.table['n_obs'].iloc[0])}, "
            f"criterion = {self.criterion})"
        )
        shown = self.table.drop(columns=["n_obs", "message"], errors="ignore")
        return f"{head}\n{'-' * len(head)}\n{shown.to_string(float_format=lambda v: f'{v:.4f}')}"

    def __repr__(self) -> str:
        return self.summary()


def select_copula(
    data: ArrayLike,
    families: str | Sequence[str] | Sequence[Copula] = "all",
    criterion: str = "aic",
    method: str = "mpl",
    gof: bool | str = False,
    k: int = 10,
    n_rep: int = 200,
    random_state: np.random.Generator | int | None = None,
    ties_method: str = "average",
) -> SelectionResult:
    r"""Fit every admissible family and rank them.

    Parameters
    ----------
    data : array_like or DataFrame
        ``(n, d)`` observations. Rank-transformed internally, so raw data is
        fine.
    families : str or sequence
        A group name (``"all"``, ``"elliptical"``, ``"archimedean"``,
        ``"extreme"``, ``"other"``, ``"baseline"``), a list of family names from
        :data:`FAMILIES`, or a list of unfitted :class:`~rcopula.core.base.Copula`
        instances when you want full control (a fixed ``df``, a particular
        ``dispstr``, a rotated family, ...).
    criterion : {"aic", "bic", "loglik", "xv"}
        Ranking column. ``"xv"`` triggers a k-fold cross-validation per family
        and is ``k`` times slower.
    method : str
        Estimation method passed to :func:`~rcopula.fit.fit`.
    gof : bool or str
        ``True`` or ``"pb"`` runs the parametric-bootstrap goodness-of-fit test
        on each family; ``"mult"`` uses the multiplier bootstrap, which is much
        faster. Adds ``gof_statistic`` and ``gof_pvalue`` columns.
    k : int
        Folds, when ``criterion="xv"``.
    n_rep : int
        Bootstrap replicates, when ``gof`` is requested.

    Returns
    -------
    SelectionResult

    Notes
    -----
    A family that fails to converge is **reported, not raised**: its row carries
    the error message and ``converged=False``, and it is ranked last. Silently
    dropping it would misrepresent the comparison.

    Examples
    --------
    The generating family is recovered:

    >>> import rcopula as rc
    >>> u = rc.ClaytonCopula(3.0).rvs(1000, random_state=0)
    >>> ranking = rc.select_copula(u, families="archimedean")
    >>> ranking.best_name
    'clayton'

    The winner comes back fitted, so it can be used immediately:

    >>> bool(abs(ranking.best.theta - 3.0) < 0.4)
    True

    The table shows *why* -- Clayton is the family with lower-tail dependence
    and no upper-tail dependence, which is what the data has:

    >>> float(ranking.table.loc["clayton", "lambda_lower"]) > 0.5
    True
    >>> float(ranking.table.loc["clayton", "lambda_upper"])
    0.0

    Upper-tail data flips the answer to Gumbel:

    >>> v = rc.GumbelCopula(2.5).rvs(1000, random_state=0)
    >>> rc.select_copula(v, families="archimedean").best_name
    'gumbel'

    Independent data prefers the parameter-free baseline, because AIC charges
    for the parameter nobody needed:

    >>> w = rc.IndependenceCopula(2).rvs(1000, random_state=0)
    >>> rc.select_copula(w, families=["independence", "gaussian", "frank"]).best_name
    'independence'
    """
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}")

    u = pseudo_obs(data, ties_method=ties_method)
    n, dim = u.shape
    candidates = _resolve(families, dim)

    gof_kind = {False: None, True: "pb"}.get(gof, gof) if isinstance(gof, bool) else gof
    if gof_kind not in (None, "pb", "mult"):
        raise ValueError(f"gof must be False, True, 'pb' or 'mult', got {gof!r}")

    rows: list[dict[str, object]] = []
    results: dict[str, CopulaFitResult] = {}

    for name, candidate in candidates:
        row: dict[str, object] = {"family": name, "n_obs": n}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = fit(candidate, u, method=method, estimate_variance=False)
        except Exception as exc:
            rows.append(
                {
                    **row,
                    "n_params": np.nan,
                    "loglik": np.nan,
                    "aic": np.nan,
                    "bic": np.nan,
                    "tau": np.nan,
                    "lambda_lower": np.nan,
                    "lambda_upper": np.nan,
                    "converged": False,
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        results[name] = res
        row.update(
            n_params=res.n_params,
            loglik=res.loglik,
            aic=res.aic,
            bic=res.bic,
            tau=_safe_scalar(res.copula.tau),
            converged=res.converged,
            message=res.message,
        )
        lower, upper = _safe_lambda(res.copula)
        row["lambda_lower"], row["lambda_upper"] = lower, upper

        if criterion == "xv":
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    row["xv"] = cross_validate(
                        candidate, u, k=k, method=method, random_state=random_state
                    )
            except Exception as exc:
                row["xv"] = np.nan
                row["message"] = f"cross-validation failed: {type(exc).__name__}: {exc}"

        if gof_kind is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    test = gof_test(
                        candidate,
                        u,
                        simulation=gof_kind,
                        estim_method=method,
                        n_rep=n_rep,
                        random_state=random_state,
                    )
                row["gof_statistic"], row["gof_pvalue"] = test.statistic, test.pvalue
            except Exception as exc:
                row["gof_statistic"], row["gof_pvalue"] = np.nan, np.nan
                row["message"] = f"gof failed: {type(exc).__name__}: {exc}"

        rows.append(row)

    table = pd.DataFrame(rows).set_index("family")
    ascending = criterion in ("aic", "bic")
    order = ["n_params", "loglik", "aic", "bic"]
    if "xv" in table:
        order.append("xv")
    if "gof_statistic" in table:
        order += ["gof_statistic", "gof_pvalue"]
    order += ["tau", "lambda_lower", "lambda_upper", "converged", "n_obs", "message"]
    table = table[[c for c in order if c in table]]
    # na_position keeps failed fits at the bottom whichever way we are sorting.
    table = table.sort_values(criterion, ascending=ascending, na_position="last")
    return SelectionResult(table=table, results=results, criterion=criterion)


def _safe_scalar(func: Callable[[], float]) -> float:
    """A dependence measure, or NaN if the family cannot report one."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(func())
    except (NotImplementedError, ValueError, ZeroDivisionError):
        return float("nan")


def _safe_lambda(copula: Copula) -> tuple[float, float]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lam = copula.lambda_()
    except (NotImplementedError, ValueError, ZeroDivisionError):
        return float("nan"), float("nan")
    return float(np.min(lam.lower)), float(np.min(lam.upper))
