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
from rcopula.core.bb import BB1Copula, BB7Copula
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
from rcopula.structural.rotated import RotatedCopula

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
    """A registry entry describing one candidate copula family for :func:`select_copula`.

    Each entry knows how to build an unfitted copula of a given dimension,
    the largest dimension the family supports, and which groups (such as
    ``"archimedean"``) it belongs to. The built-in entries live in
    :data:`FAMILIES`; you rarely need to create one yourself.

    Parameters
    ----------
    name : str
        Key in :data:`FAMILIES`, e.g. ``"clayton"``.
    factory : callable
        Function ``factory(dim: int) -> Copula`` returning an unfitted
        instance of the family in dimension ``dim``.
    max_dim : int or None
        Largest supported dimension; ``None`` means any dimension.
    groups : frozenset of str
        Group labels accepted by ``select_copula(families=...)``.

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

    Examples
    --------
    >>> from rcopula.select import FAMILIES
    >>> spec = FAMILIES["tawn"]
    >>> spec.max_dim, spec.admissible(2), spec.admissible(3)
    (2, True, False)
    """

    name: str
    factory: Callable[[int], Copula]
    max_dim: int | None
    groups: frozenset[str]

    def admissible(self, dim: int) -> bool:
        """Check whether this family can be used for data with ``dim`` columns.

        Parameters
        ----------
        dim : int
            Number of variables (columns) in the data, at least 2.

        Returns
        -------
        bool
            ``True`` if ``max_dim`` is ``None`` or ``dim <= max_dim``.
        """
        return self.max_dim is None or dim <= self.max_dim


def _spec(
    name: str,
    factory: Callable[[int], Copula],
    *groups: str,
    max_dim: int | None = None,
    in_all: bool = True,
) -> FamilySpec:
    return FamilySpec(name, factory, max_dim, frozenset(groups) | ({"all"} if in_all else set()))


def _rotated(base: Callable[[int], Copula], degrees: int) -> Callable[[int], Copula]:
    """Factory for a rotation of a bivariate family (or the survival copula at 180)."""

    def build(dim: int) -> Copula:
        if degrees == 180:
            return RotatedCopula(base(dim), True)
        return RotatedCopula(base(dim), degrees)

    return build


def _rotations() -> list[FamilySpec]:
    """The 90/180/270-degree rotations of the one-sided families, and BB1/BB7.

    Kept out of the ``"all"`` group: they are mainly pair-copulas for vines,
    and adding 20 candidates to every ``select_copula(u)`` call would slow it
    down and change what it has always returned. Ask for them by name, or by
    the ``"rotated"``, ``"bb"`` or ``"vine"`` groups.
    """
    specs = [
        _spec("bb1", lambda d: BB1Copula(dim=d), "bb", "vine", max_dim=2, in_all=False),
        _spec("bb7", lambda d: BB7Copula(dim=d), "bb", "vine", max_dim=2, in_all=False),
    ]
    bases: dict[str, Callable[[int], Copula]] = {
        "clayton": lambda d: ClaytonCopula(dim=d),
        "gumbel": lambda d: GumbelCopula(dim=d),
        "joe": lambda d: JoeCopula(dim=d),
        "bb1": lambda d: BB1Copula(dim=d),
        "bb7": lambda d: BB7Copula(dim=d),
    }
    for base, factory in bases.items():
        sep = "_" if base.startswith("bb") else ""
        groups = ("rotated", "vine") + (("bb",) if base.startswith("bb") else ())
        for degrees in (90, 180, 270):
            survival_any_dim = degrees == 180 and not base.startswith("bb")
            specs.append(
                _spec(
                    f"{base}{sep}{degrees}",
                    _rotated(factory, degrees),
                    *groups,
                    max_dim=None if survival_any_dim else 2,
                    in_all=False,
                )
            )
    return specs


#: The candidate registry. Keys are the names used in ``families=[...]`` and in
#: the result table; group labels (``"elliptical"``, ``"archimedean"``,
#: ``"extreme"``, ``"all"``) select several at once.
#:
#: Families excluded on purpose: Marshall-Olkin (its density is undefined on a
#: curve, so the likelihood is not comparable), the Frechet bounds (no density),
#: and FGM beyond ``d = 2`` (``2^d - d - 1`` parameters, 1013 at ``d = 10``).
#:
#: The rotations (``"clayton90"``, ``"clayton180"``, ``"clayton270"``, the same
#: for ``gumbel`` and ``joe``, and ``"bb1_90"`` ... ``"bb7_270"``) and the
#: two-parameter ``"bb1"`` and ``"bb7"`` are registered too, but outside the
#: ``"all"`` group: they are chiefly vine pair-copulas, chosen through the
#: ``"rotated"``, ``"bb"`` and ``"vine"`` groups or by name. Degrees follow
#: :class:`~rcopula.RotatedCopula`: 90 reflects the second argument, 270 the
#: first, 180 both (the survival copula). R ``VineCopula``'s family codes
#: 2x/3x are the other way round -- its 90 (e.g. 23) reflects the first
#: argument, so it is rcopula's 270.
FAMILIES: dict[str, FamilySpec] = {
    spec.name: spec
    for spec in (
        _spec("independence", lambda d: IndependenceCopula(d), "baseline", "vine"),
        _spec("gaussian", lambda d: GaussianCopula(dim=d), "elliptical", "vine"),
        _spec("student", lambda d: StudentCopula(dim=d), "elliptical", "vine"),
        _spec("clayton", lambda d: ClaytonCopula(dim=d), "archimedean", "vine"),
        _spec("gumbel", lambda d: GumbelCopula(dim=d), "archimedean", "extreme", "vine"),
        _spec("frank", lambda d: FrankCopula(dim=d), "archimedean", "vine"),
        _spec("joe", lambda d: JoeCopula(dim=d), "archimedean", "vine"),
        _spec("amh", lambda d: AMHCopula(dim=d), "archimedean"),
        _spec("galambos", lambda d: GalambosCopula(1.0), "extreme", max_dim=2),
        _spec("husler_reiss", lambda d: HuslerReissCopula(1.0), "extreme", max_dim=2),
        _spec("tawn", lambda d: TawnCopula(0.5), "extreme", max_dim=2),
        _spec("tev", lambda d: TEVCopula(0.5), "extreme", max_dim=2),
        _spec("plackett", lambda d: PlackettCopula(2.0), "other", max_dim=2),
        _spec("fgm", lambda d: FGMCopula(0.3), "other", max_dim=2),
        *_rotations(),
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
    r"""Score a copula family by how well it predicts data it was not fitted on (higher is better).

    Splits the data into ``k`` parts, fits on all but one part and scores
    the left-out part, rotating through every part. Use it to compare
    families more honestly than with AIC, at the cost of ``k`` fits per
    family. This is R's ``xvCopula``.

    Parameters
    ----------
    copula : Copula
        Family to score; its parameter values are only a starting point.
    data : array_like of float or pandas.DataFrame, shape (n, d)
        Observations, one row per observation; ``d`` must equal
        ``copula.dim``. Rank-transformed internally, so raw data is fine.
    k : int, default 10
        Number of folds, between 2 and ``n``.
    method : {"mpl", "ml", "itau", "irho", "itau.mpl"}, default "mpl"
        Estimation method for each training fit; see :func:`~rcopula.fit`.
    random_state : int, numpy.random.Generator or None, default None
        Controls the random fold assignment. Pass an int for reproducible
        results.
    ties_method : {"average", "min", "max", "dense", "ordinal", "random"}, default "average"
        How tied values are ranked when ``data`` is turned into
        pseudo-observations.

    Returns
    -------
    float
        Cross-validated log-likelihood, rescaled to a full-sample scale;
        higher is better.

    Raises
    ------
    ValueError
        If ``k`` is not between 2 and the number of observations, or if a
        training fit fails (for example, the data have the wrong number of
        columns).

    Notes
    -----
    Fits on ``k-1`` folds and scores the held-out one, summed over folds and
    **multiplied by** :math:`n/(n - n/k)` so the result is on the scale of a
    full-sample log-likelihood and comparable across families -- the same
    convention R uses.

    This is the honest answer to a real problem: the ordinary AIC is biased for
    copulas, because the pseudo-observations are themselves estimated from the
    data and the usual "one penalty unit per parameter" accounting no longer
    holds (Gronneberg & Hjort 2014). Cross-validation sidesteps the bias by
    scoring on data the fit never saw. It costs ``k`` fits per family.

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
    """The ranking from :func:`select_copula`: a comparison table plus the winning fitted copula.

    Print it to see the table; use :attr:`best` to get the winning copula,
    already fitted, and :attr:`results` for every candidate's full fit.

    Parameters
    ----------
    table : pandas.DataFrame
        Ranked comparison table (see Attributes).
    results : dict of str to CopulaFitResult
        Fit result per family name.
    criterion : {"aic", "bic", "loglik", "xv"}
        Column that decided the ranking.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per candidate, indexed by family name and sorted best-first.
        Columns: ``n_params``, ``loglik``, ``aic``, ``bic``, ``tau``,
        ``lambda_lower``, ``lambda_upper``, ``converged``, ``n_obs``,
        ``message``, plus ``xv`` and the goodness-of-fit columns
        (``gof_statistic``, ``gof_pvalue``) when those were requested.
    results : dict of str to CopulaFitResult
        Family name to :class:`~rcopula.fit.results.CopulaFitResult`, including
        the ones that did not converge (absent if the fit raised).
    criterion : str
        Which column decided the ranking.
    """

    table: pd.DataFrame
    results: dict[str, CopulaFitResult]
    criterion: str

    @property
    def best_name(self) -> str:
        """The name of the family that ranked first, e.g. ``"clayton"``.

        Returns
        -------
        str

        Raises
        ------
        ValueError
            If the table is empty.
        """
        if self.table.empty:
            raise ValueError("no family could be fitted")
        return str(self.table.index[0])

    @property
    def best_result(self) -> CopulaFitResult:
        """The full fit result (estimates, log-likelihood, ...) of the winning family.

        Returns
        -------
        CopulaFitResult

        Raises
        ------
        ValueError
            If the table is empty.
        KeyError
            If the top-ranked family has no fit result because its fit
            raised (possible only when every candidate failed).
        """
        return self.results[self.best_name]

    @property
    def best(self) -> Copula:
        """The winning copula, with its fitted parameters, ready to sample from or evaluate.

        Returns
        -------
        Copula
        """
        return self.best_result.copula

    def summary(self) -> str:
        """The ranking as a human-readable text table, ready to print.

        Includes a header with the sample size and criterion; the ``n_obs``
        and ``message`` columns are left out for width. ``repr`` of the
        result shows the same text.

        Returns
        -------
        str
            Multi-line text; pass it to ``print``.
        """
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
    r"""Try many copula families on your data and rank them from best to worst fit.

    Fits every candidate family that supports your data's dimension, scores
    each one (by AIC unless you choose otherwise), and returns a sorted
    table along with the winning copula, already fitted. Use it when you do
    not know which family to pick.

    Parameters
    ----------
    data : array_like of float or pandas.DataFrame, shape (n, d)
        Observations, one row per observation and one column per variable.
        Rank-transformed internally, so raw data is fine.
    families : str, sequence of str or sequence of Copula, default "all"
        Which candidates to try: a group name (``"all"``, ``"elliptical"``,
        ``"archimedean"``, ``"extreme"``, ``"other"``, ``"baseline"``, or
        ``"rotated"``, ``"bb"`` and ``"vine"`` for the rotations and BB1/BB7,
        which ``"all"`` leaves out), a list of family names from
        :data:`FAMILIES` (e.g. ``["clayton", "clayton270", "bb1"]``), or a
        list of unfitted :class:`~rcopula.core.base.Copula` instances when you
        want full control (a fixed ``df``, a particular ``dispstr``, ...).
        Families limited to two dimensions are skipped automatically for a
        group name.
    criterion : {"aic", "bic", "loglik", "xv"}, default "aic"
        Ranking column. ``"aic"`` and ``"bic"`` rank lowest first; ``"loglik"``
        and ``"xv"`` rank highest first. ``"xv"`` triggers a k-fold
        cross-validation per family and is ``k`` times slower.
    method : {"mpl", "ml", "itau", "irho", "itau.mpl"}, default "mpl"
        Estimation method passed to :func:`~rcopula.fit.fit`.
    gof : bool or {"pb", "mult"}, default False
        ``True`` or ``"pb"`` runs the parametric-bootstrap goodness-of-fit test
        on each family; ``"mult"`` uses the multiplier bootstrap, which is much
        faster. Adds ``gof_statistic`` and ``gof_pvalue`` columns.
    k : int, default 10
        Number of folds (2 to ``n``), used only when ``criterion="xv"``.
    n_rep : int, default 200
        Bootstrap replicates (positive), used only when ``gof`` is requested.
    random_state : int, numpy.random.Generator or None, default None
        Seed for the cross-validation folds and the goodness-of-fit bootstrap.
        Every family gets the *same* folds and the same bootstrap stream, so
        the comparison is not confounded by random draws: an int is used as
        the seed for each family, and a Generator (or ``None``) is reduced to
        a single int seed drawn once (one draw from the Generator, and only
        when ``criterion="xv"`` or ``gof`` is requested). Pass an int for
        reproducible results.
    ties_method : {"average", "min", "max", "dense", "ordinal", "random"}, default "average"
        How tied values are ranked when ``data`` is turned into
        pseudo-observations.

    Returns
    -------
    SelectionResult
        The ranked table (``.table``), the winner (``.best``,
        ``.best_name``) and every fit (``.results``).

    Raises
    ------
    ValueError
        If ``criterion`` or ``gof`` is not one of the listed values, a family
        name or group is unknown, a named family does not support the data's
        dimension, ``families`` is empty, or a supplied copula's ``dim`` does
        not match the data.
    TypeError
        If ``families`` mixes names and copula objects, or contains something
        that is neither.

    Notes
    -----
    A family whose fit **raises** is reported, not raised: its row carries
    the error message, ``NaN`` scores and ``converged=False``, and it is
    ranked last. A fit that runs but whose optimiser reports non-convergence
    keeps its score and is ranked by it, with ``converged=False`` in its
    row. Silently dropping either would misrepresent the comparison. If both
    the cross-validation and the goodness-of-fit run fail for a family, the
    ``message`` column carries both errors, separated by ``"; "``.

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

    On independent data no candidate separates from the parameter-free baseline.
    Twice the log-likelihood is the likelihood-ratio statistic against
    independence, and for a one-parameter family it stays below the 0.1% point
    of a chi-squared with one degree of freedom. *Which* family tops the table
    there is a coin flip, and reading it as a result would be a mistake:

    >>> w = rc.IndependenceCopula(2).rvs(2000, random_state=0)
    >>> table = rc.select_copula(w, families=["independence", "gaussian", "frank"]).table
    >>> bool((2 * table["loglik"] < 10.83).all())
    True

    Dependence of any real size settles it immediately:

    >>> x = rc.FrankCopula(4.0).rvs(2000, random_state=0)
    >>> table = rc.select_copula(x, families=["independence", "gaussian", "frank"]).table
    >>> bool(table.loc["independence", "aic"] - table["aic"].min() > 100)
    True
    """
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}")

    u = pseudo_obs(data, ties_method=ties_method)
    n, dim = u.shape
    candidates = _resolve(families, dim)

    gof_kind = {False: None, True: "pb"}.get(gof, gof) if isinstance(gof, bool) else gof
    if gof_kind not in (None, "pb", "mult"):
        raise ValueError(f"gof must be False, True, 'pb' or 'mult', got {gof!r}")

    # Every family must see the same folds and the same bootstrap stream, or the
    # comparison is partly a comparison of random draws. An int seed already
    # gives that (each call re-seeds from it); a Generator, or None, would be
    # consumed family after family, so it is reduced to one int seed up front.
    # The generator is touched only when something random is actually run.
    family_seed: int | None = None
    if criterion == "xv" or gof_kind is not None:
        if isinstance(random_state, np.random.Generator):
            family_seed = int(random_state.integers(0, 2**63 - 1))
        elif random_state is None:
            family_seed = int(np.random.default_rng().integers(0, 2**63 - 1))
        else:
            family_seed = int(random_state)

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
        failures: list[str] = []

        if criterion == "xv":
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    row["xv"] = cross_validate(
                        candidate, u, k=k, method=method, random_state=family_seed
                    )
            except Exception as exc:
                row["xv"] = np.nan
                failures.append(f"cross-validation failed: {type(exc).__name__}: {exc}")

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
                        random_state=family_seed,
                    )
                row["gof_statistic"], row["gof_pvalue"] = test.statistic, test.pvalue
            except Exception as exc:
                row["gof_statistic"], row["gof_pvalue"] = np.nan, np.nan
                failures.append(f"gof failed: {type(exc).__name__}: {exc}")

        if failures:
            # Keep every failure: a later one must not hide an earlier one.
            row["message"] = "; ".join(failures)
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
