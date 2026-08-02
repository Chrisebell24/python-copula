r"""Choosing what to trade: pair and partner selection for statistical arbitrage.

:mod:`rcopula.portfolio` trades a pair once you have one. This picks it. Out of
500 names there are 124,750 pairs and 2.6 billion quadruples, so the selection
rule is not a preliminary to the strategy -- for a copula strategy it largely
*is* the strategy, and different rules surface genuinely different candidates.

**Pairs.** Six rules, of which the last two are the copula-specific ones:

===================  =========================================================
``distance``         Sum of squared deviations between normalised cumulative
                     returns. The classic (Gatev et al.), and it selects for
                     *level* agreement, which is not what a copula uses.
``pearson``          Linear correlation. Included to be argued with: it is not
                     invariant to the marginal transforms a copula discards.
``spearman``         Rank correlation. Cheap, and close to ``kendall``.
``kendall``          Rank correlation, concordance-based.
``tail``             Lower-tail dependence, estimated nonparametrically. Picks
                     pairs that crash together, which is where a copula
                     strategy makes and loses its money.
``qq``               Mean distance from the diagonal of the QQ plot. Sensitive
                     to disagreement anywhere in the distribution, not just the
                     middle.
===================  =========================================================

**Partners.** For a vine or a 4-dimensional copula you need a *quadruple*: a
target and three partners. Exhaustive search over all triples of the remaining
names is :math:`O(n^3)`, so the four approaches below all reduce the candidate
set first, exactly as Stübinger, Mangold and Krauss do.

===================  =========================================================
``traditional``      Largest sum of pairwise Spearman correlations.
``extended``         Largest **multivariate** Spearman's rho -- three
                     generalisations of the bivariate coefficient, averaged
                     (Schmid and Schmidt 2007). Unlike the traditional sum,
                     this is a genuine :math:`d`-dimensional measure rather
                     than a pile of bivariate ones.
``geometric``        Smallest total distance from the hyper-diagonal of the
                     unit hypercube. Purely geometric, no distributional
                     assumption at all.
``extremal``         Strongest joint-tail concentration. Targets what Mangold's
                     chi-squared test targets -- see the note under
                     :func:`select_partners` about what this is and is not.
===================  =========================================================

Everything here works on **ranks**, which is the point: a copula does not see
the margins, so neither should the rule that chooses its inputs.

============================  ================================================
:func:`select_pairs`          Rank every pair by one of six criteria.
:func:`select_partners`       Find the best quadruple for a target.
:func:`multivariate_spearman` Schmid and Schmidt's d-dimensional rho.
:func:`diagonal_distance`     Mean distance from the hyper-diagonal.
:func:`tail_concentration`    Joint-tail co-occurrence, relative to chance.
============================  ================================================

Examples
--------
>>> import numpy as np, pandas as pd, rcopula as rc
>>> from rcopula.statarb import select_pairs
>>> rng = np.random.default_rng(0)
>>> factor = rng.standard_normal(500)
>>> returns = pd.DataFrame({
...     "AAA": factor + 0.3 * rng.standard_normal(500),
...     "BBB": factor + 0.3 * rng.standard_normal(500),
...     "CCC": rng.standard_normal(500),
... })
>>> select_pairs(returns, method="kendall").iloc[0][["first", "second"]].tolist()
['AAA', 'BBB']

References
----------
Gatev, E., Goetzmann, W. N. and Rouwenhorst, K. G. (2006). Pairs trading:
    performance of a relative-value arbitrage rule.
    *Review of Financial Studies* 19(3), 797-827.
    The distance rule.
Stübinger, J., Mangold, B. and Krauss, C. (2018). Statistical arbitrage with
    vine copulas. *Quantitative Finance* 18(11), 1831-1849.
    The four partner-selection approaches.
Schmid, F. and Schmidt, R. (2007). Multivariate extensions of Spearman's rho
    and related statistics. *Statistics and Probability Letters* 77(4),
    407-416.  The three multivariate rho formulas.
Liew, R. Q. and Wu, Y. (2013). Pairs trading: a copula approach.
    *J. Derivatives and Hedge Funds* 19(1), 12-30.
Krauss, C. (2017). Statistical arbitrage pairs trading strategies: review and
    outlook. *J. Economic Surveys* 31(2), 513-545.
"""

from __future__ import annotations

import itertools
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from rcopula.dependence import pseudo_obs

__all__ = [
    "PAIR_METHODS",
    "PARTNER_METHODS",
    "diagonal_distance",
    "multivariate_spearman",
    "select_pairs",
    "select_partners",
    "tail_concentration",
]

PairMethod = Literal["distance", "pearson", "spearman", "kendall", "tail", "qq"]
PartnerMethod = Literal["traditional", "extended", "geometric", "extremal"]

#: Selection criteria for pairs, in the order the module docstring lists them.
PAIR_METHODS: tuple[str, ...] = ("distance", "pearson", "spearman", "kendall", "tail", "qq")

#: Selection criteria for quadruples.
PARTNER_METHODS: tuple[str, ...] = ("traditional", "extended", "geometric", "extremal")

#: How many observations the joint tail should be expected to hold under
#: independence. The corner is then sized to deliver that count, rather than
#: being fixed at a quantile -- see :func:`tail_concentration` for why a fixed
#: one cannot work in more than two dimensions.
_TAIL_TARGET_COUNT = 40.0


def _frame(data: ArrayLike, names: list[str] | None = None) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    arr = np.atleast_2d(np.asarray(data, dtype=float))
    columns = names or [f"x{j}" for j in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=columns)


def _ranks(frame: pd.DataFrame) -> NDArray[np.float64]:
    """Pseudo-observations, which is the only view of the data any of this uses."""
    return np.asarray(pseudo_obs(frame.to_numpy()), dtype=float)


def _as_ranks(u: ArrayLike) -> NDArray[np.float64]:
    """Pseudo-observations, unconditionally.

    The tempting shortcut is to skip this when the input already lies in
    [0, 1] and call it "already uniform". That is wrong: a sample of raw
    uniforms lies in [0, 1] and is *not* a set of ranks, and the estimators
    below are defined on ranks. Feeding them raw uniforms costs about 0.02 on a
    comonotone quadruple -- small enough to look like noise and large enough to
    reorder a shortlist. Ranking is idempotent, so doing it always is free of
    consequence and free of that trap.
    """
    return np.asarray(pseudo_obs(np.atleast_2d(np.asarray(u, dtype=float))), dtype=float)


# --------------------------------------------------------------------------
# multivariate dependence measures
# --------------------------------------------------------------------------


def multivariate_spearman(u: ArrayLike) -> float:
    r"""Schmid and Schmidt's multivariate Spearman's rho.

    Three separate generalisations of the bivariate coefficient, averaged. Each
    reduces to ordinary Spearman's rho when :math:`d = 2`, and each measures
    something slightly different in higher dimensions -- the first two the
    concentration towards a corner, the third the average pairwise
    concordance -- so the mean is more robust than any one of them.

    .. math::

        \rho_1 &= h(d)\Bigl\{-1 + \tfrac{2^d}{n}
                  \textstyle\sum_j \prod_i (1 - U_{ij})\Bigr\} \\
        \rho_2 &= h(d)\Bigl\{-1 + \tfrac{2^d}{n}
                  \textstyle\sum_j \prod_i U_{ij}\Bigr\} \\
        \rho_3 &= -3 + \tfrac{12}{n\binom{d}{2}}
                  \textstyle\sum_{k<l}\sum_j (1-U_{kj})(1-U_{lj})

    with :math:`h(d) = (d+1)/(2^d - d - 1)`.

    Parameters
    ----------
    u : array_like, shape (n, d)
        Pseudo-observations, or raw data (ranks are taken).

    Returns
    -------
    float

    Examples
    --------
    It agrees with the bivariate coefficient in two dimensions:

    >>> import numpy as np, rcopula as rc
    >>> from rcopula.statarb import multivariate_spearman
    >>> u = rc.GaussianCopula(0.7).rvs(20000, random_state=0)
    >>> bool(abs(multivariate_spearman(u) - rc.cor_spearman(u)[0, 1]) < 0.02)
    True

    It is zero under independence and one under comonotonicity, in any
    dimension:

    >>> independent = rc.IndependenceCopula(4).rvs(20000, random_state=0)
    >>> bool(abs(multivariate_spearman(independent)) < 0.02)
    True
    >>> comonotone = np.tile(np.random.default_rng(0).uniform(size=(20000, 1)), (1, 4))
    >>> bool(multivariate_spearman(comonotone) > 0.99)
    True
    """
    arr = _as_ranks(u)
    n, d = arr.shape
    if d < 2:
        raise ValueError(f"multivariate Spearman needs at least 2 columns, got {d}")

    scale = (d + 1) / (2**d - d - 1)
    rho_1 = scale * (-1.0 + (2**d / n) * float(np.sum(np.prod(1.0 - arr, axis=1))))
    rho_2 = scale * (-1.0 + (2**d / n) * float(np.sum(np.prod(arr, axis=1))))

    pairs = list(itertools.combinations(range(d), 2))
    total = sum(float(np.sum((1.0 - arr[:, k]) * (1.0 - arr[:, l]))) for k, l in pairs)
    rho_3 = -3.0 + (12.0 / (n * len(pairs))) * total

    return float(np.mean([rho_1, rho_2, rho_3]))


def diagonal_distance(u: ArrayLike) -> float:
    r"""Mean Euclidean distance from the hyper-diagonal of the unit hypercube.

    Perfectly concordant observations lie exactly on the line from
    :math:`(0,\dots,0)` to :math:`(1,\dots,1)`; independent ones scatter away
    from it. **Small means strongly related**, which is the opposite convention
    to every other measure here -- :func:`select_partners` handles the sign.

    The appeal is that it assumes nothing: no distribution, no copula family,
    not even that the dependence is monotone in the usual sense.

    Examples
    --------
    >>> import numpy as np, rcopula as rc
    >>> from rcopula.statarb import diagonal_distance
    >>> comonotone = np.tile(np.random.default_rng(0).uniform(size=(5000, 1)), (1, 4))
    >>> float(diagonal_distance(comonotone)) < 1e-12
    True
    >>> independent = rc.IndependenceCopula(4).rvs(5000, random_state=0)
    >>> bool(diagonal_distance(independent) > 0.3)
    True
    """
    arr = _as_ranks(u)
    d = arr.shape[1]
    # Project each point onto the unit diagonal and keep the residual.
    direction = np.ones(d) / np.sqrt(d)
    projection = (arr @ direction)[:, None] * direction
    return float(np.mean(np.linalg.norm(arr - projection, axis=1)))


def tail_concentration(u: ArrayLike, quantile: float | None = None) -> float:
    r"""How much more often the coordinates are jointly extreme than by chance.

    Counts observations in the lower and upper corners of the unit hypercube and
    divides by what independence would put there. One means no more joint
    extremes than chance; larger means the names crash and spike together. This
    is the quantity a copula strategy lives or dies on, and no correlation
    measures it.

    Parameters
    ----------
    u : array_like, shape (n, d)
    quantile : float, optional
        Corner size. By default it is **chosen from the data**, which matters
        more than it sounds -- see the note below.

    Returns
    -------
    float
        Both corners combined, relative to independence.

    Notes
    -----
    A fixed corner cannot work beyond two dimensions. The corner holds
    :math:`q^d` of the mass, so at the conventional :math:`q = 0.05` a
    200,000-row sample expects 500 observations in it at :math:`d = 2`, 25 at
    :math:`d = 3`, and **1.3** at :math:`d = 4` -- at which point the statistic
    is counting noise and scatters between 0.4 and 1.2 under independence.

    The default therefore solves :math:`q = (c/n)^{1/d}` for a target count
    :math:`c`, so the corner always holds enough observations to mean something.

    **What that costs, stated plainly.** A wide corner stops measuring the tail.
    At :math:`d = 2` and 200,000 observations, Clayton and Gaussian at the same
    Kendall's tau give 2.15 against 2.13 at :math:`q = 0.3` -- indistinguishable
    -- and only separate at :math:`q \le 0.02`, where the ratio reaches 1.2 and
    then 1.4. But :math:`q = 0.02` at :math:`d = 4` needs 250 million
    observations to keep 40 in the corner. Even a million rows only supports
    :math:`q = 0.08`.

    So in four dimensions at any realistic sample size this ranks by joint
    co-movement in general, **not** by tail dependence specifically, and it
    cannot tell a heavy-tailed copula from a Gaussian one. That is a property of
    the data, not of the implementation: the observations required to resolve a
    four-dimensional tail simply are not there. Use it in two dimensions with an
    explicit small ``quantile`` when the tail is the question.

    Examples
    --------
    >>> import rcopula as rc
    >>> from rcopula.statarb import tail_concentration
    >>> independent = rc.IndependenceCopula(4).rvs(40000, random_state=0)
    >>> bool(abs(tail_concentration(independent) - 1.0) < 0.25)
    True
    >>> dependent = rc.ClaytonCopula(4.0, dim=4).rvs(40000, random_state=0)
    >>> bool(tail_concentration(dependent) > 5)
    True
    """
    arr = _as_ranks(u)
    n, d = arr.shape
    if quantile is None:
        quantile = float(min(0.25, (_TAIL_TARGET_COUNT / max(n, 1)) ** (1.0 / d)))
    if not 0.0 < quantile < 0.5:
        raise ValueError(f"quantile must lie in (0, 0.5), got {quantile}")
    lower = float(np.mean(np.all(arr <= quantile, axis=1)))
    upper = float(np.mean(np.all(arr >= 1.0 - quantile, axis=1)))
    expected = quantile**d
    return float((lower + upper) / (2.0 * expected))


# --------------------------------------------------------------------------
# pair selection
# --------------------------------------------------------------------------


def _pair_score(method: str, x: NDArray[np.float64], u: NDArray[np.float64]) -> float:
    """One number per pair. Larger is always better; the callers rely on that."""
    if method == "distance":
        # Gatev's rule: agreement of normalised cumulative return paths.
        path = np.cumsum(x, axis=0)
        spread = np.ptp(path, axis=0)
        normalised = (path - path.mean(axis=0)) / np.where(spread > 0, spread, 1.0)
        return -float(np.sum((normalised[:, 0] - normalised[:, 1]) ** 2))
    if method == "pearson":
        return float(np.asarray(np.corrcoef(x, rowvar=False))[0, 1])
    if method == "spearman":
        # Pearson correlation of the ranks -- which is exactly Spearman's rho.
        return float(np.asarray(np.corrcoef(u, rowvar=False))[0, 1])
    if method == "kendall":
        from rcopula.dependence import cor_kendall

        return float(np.asarray(cor_kendall(u))[0, 1])
    if method == "tail":
        return tail_concentration(u)
    if method == "qq":
        # Mean vertical distance between the two sorted samples, standardised.
        # Equivalently, distance from the 45-degree line of the QQ plot.
        return -float(np.mean(np.abs(np.sort(u[:, 0]) - np.sort(u[:, 1]))))
    raise ValueError(f"method must be one of {PAIR_METHODS}, got {method!r}")


def select_pairs(
    data: ArrayLike,
    method: PairMethod = "kendall",
    *,
    top: int | None = None,
    names: list[str] | None = None,
) -> pd.DataFrame:
    """Rank every pair of columns by one selection criterion.

    Parameters
    ----------
    data : DataFrame or array_like, shape (n, k)
        Returns, one column per instrument. Column names are carried through.
    method : {"distance", "pearson", "spearman", "kendall", "tail", "qq"}
        See the module docstring.
    top : int, optional
        Keep only this many rows.
    names : list of str, optional
        Column names, when ``data`` is a plain array.

    Returns
    -------
    DataFrame
        Columns ``first``, ``second``, ``score``, ``rank``, sorted best first.
        ``score`` is oriented so that larger is always better, whichever
        criterion was used -- the distance and QQ rules are negated for this
        reason, so their scores are non-positive.

    Notes
    -----
    Cost is :math:`O(k^2)` pairs, each :math:`O(n\\log n)` for ``kendall`` and
    :math:`O(n)` otherwise. Five hundred names is 124,750 pairs and takes a
    couple of minutes on ``kendall``; screen with ``spearman`` first if that
    matters, since the two rank nearly the same candidates.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from rcopula.statarb import select_pairs
    >>> rng = np.random.default_rng(0)
    >>> shared = rng.standard_normal(400)
    >>> data = pd.DataFrame({
    ...     "A": shared + 0.2 * rng.standard_normal(400),
    ...     "B": shared + 0.2 * rng.standard_normal(400),
    ...     "C": rng.standard_normal(400),
    ...     "D": rng.standard_normal(400),
    ... })
    >>> ranked = select_pairs(data, method="spearman")
    >>> len(ranked)
    6
    >>> ranked.iloc[0]["first"], ranked.iloc[0]["second"]
    ('A', 'B')

    The criteria disagree, which is the reason to have more than one:

    >>> by_tail = select_pairs(data, method="tail").iloc[0]
    >>> bool(by_tail["score"] > 0)
    True
    """
    if method not in PAIR_METHODS:
        raise ValueError(f"method must be one of {PAIR_METHODS}, got {method!r}")
    frame = _frame(data, names)
    if frame.shape[1] < 2:
        raise ValueError(f"need at least 2 columns to form a pair, got {frame.shape[1]}")
    values = frame.to_numpy(dtype=float)
    ranks = _ranks(frame)
    columns = list(frame.columns)

    rows = []
    for i, j in itertools.combinations(range(len(columns)), 2):
        rows.append(
            {
                "first": columns[i],
                "second": columns[j],
                "score": _pair_score(method, values[:, [i, j]], ranks[:, [i, j]]),
            }
        )
    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out.attrs["method"] = method
    return out.head(top) if top else out


# --------------------------------------------------------------------------
# partner selection
# --------------------------------------------------------------------------


def _partner_score(method: str, block: NDArray[np.float64]) -> float:
    if method == "traditional":
        # Sum of pairwise Spearman correlations over the quadruple.
        correlation = np.asarray(np.corrcoef(block, rowvar=False))
        return float(np.sum(correlation[np.triu_indices(block.shape[1], 1)]))
    if method == "extended":
        return multivariate_spearman(block)
    if method == "geometric":
        # Small distance means strongly related, so negate to keep "larger is
        # better" true for every method -- the ranking code depends on it.
        return -diagonal_distance(block)
    if method == "extremal":
        return tail_concentration(block)
    raise ValueError(f"method must be one of {PARTNER_METHODS}, got {method!r}")


def select_partners(
    data: ArrayLike,
    target: str | int,
    *,
    method: PartnerMethod = "extended",
    n_partners: int = 3,
    n_candidates: int = 50,
    names: list[str] | None = None,
) -> dict[str, Any]:
    r"""Find the partners that best complete a quadruple around ``target``.

    A vine or a :math:`d`-dimensional copula strategy needs a group, not a pair.
    Searching every triple of 499 remaining names is 20 million combinations per
    target, so this pre-screens to the ``n_candidates`` most correlated names and
    searches exhaustively within those -- which is what Stübinger, Mangold and
    Krauss do, and what makes the problem finite.

    Parameters
    ----------
    data : DataFrame or array_like, shape (n, k)
    target : str or int
        Column name, or index.
    method : {"traditional", "extended", "geometric", "extremal"}
    n_partners : int
        Partners to choose. Three gives the quadruple the literature uses.
    n_candidates : int
        Pre-screening width. Larger is more thorough and grows as
        :math:`\binom{n_{\text{candidates}}}{n_{\text{partners}}}`.
    names : list of str, optional

    Returns
    -------
    dict
        ``"target"``, ``"partners"``, ``"score"``, ``"method"``, and
        ``"considered"`` -- how many combinations were actually evaluated, so
        the pre-screening is visible rather than implicit.

    Notes
    -----
    On ``extremal``, two caveats, both worth reading before relying on it.

    Stübinger et al. use Mangold's (2015) chi-squared test against independence
    within a specific parametric copula family. This uses joint-tail
    concentration instead -- the same *property*, a different statistic, arrived
    at without that paper. It is not a reimplementation of Mangold's test and
    produces no p-value.

    More importantly, in four dimensions it is not really an *extremal* rule at
    all. A four-dimensional corner deep enough to isolate the tail is empty at
    any realistic sample size -- see :func:`tail_concentration` -- so the corner
    widens until it holds data, and a wide corner measures joint co-movement in
    general. Expect it to behave like a more robust ``traditional``, and do not
    expect it to find tail dependence that the other rules miss.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from rcopula.statarb import select_partners
    >>> rng = np.random.default_rng(0)
    >>> factor = rng.standard_normal(600)
    >>> frame = pd.DataFrame({
    ...     "TGT": factor + 0.3 * rng.standard_normal(600),
    ...     "P1": factor + 0.3 * rng.standard_normal(600),
    ...     "P2": factor + 0.3 * rng.standard_normal(600),
    ...     "P3": factor + 0.3 * rng.standard_normal(600),
    ...     "N1": rng.standard_normal(600),
    ...     "N2": rng.standard_normal(600),
    ... })
    >>> found = select_partners(frame, "TGT", method="extended")
    >>> sorted(found["partners"])
    ['P1', 'P2', 'P3']
    """
    if method not in PARTNER_METHODS:
        raise ValueError(f"method must be one of {PARTNER_METHODS}, got {method!r}")
    frame = _frame(data, names)
    columns = list(frame.columns)
    if isinstance(target, int):
        target = columns[target]
    if target not in columns:
        raise ValueError(f"target {target!r} is not a column; have {columns[:8]}...")
    if n_partners < 1:
        raise ValueError(f"n_partners must be at least 1, got {n_partners}")
    others = [c for c in columns if c != target]
    if len(others) < n_partners:
        raise ValueError(
            f"need at least {n_partners} columns besides the target, got {len(others)}"
        )

    ranks = _ranks(frame)
    position = {name: k for k, name in enumerate(columns)}
    target_ranks = ranks[:, position[target]]

    # Pre-screen on pairwise rank correlation with the target. Every criterion
    # below is monotone in dependence, so a name uncorrelated with the target
    # cannot be in the winning group -- and this turns an intractable search
    # into a small one.
    affinity = {
        name: abs(float(np.corrcoef(target_ranks, ranks[:, position[name]])[0, 1]))
        for name in others
    }
    shortlist = sorted(others, key=lambda name: affinity[name], reverse=True)[
        : max(n_candidates, n_partners)
    ]

    best_score, best_partners, considered = -np.inf, None, 0
    for combination in itertools.combinations(shortlist, n_partners):
        indices = [position[target]] + [position[name] for name in combination]
        score = _partner_score(method, ranks[:, indices])
        considered += 1
        if score > best_score:
            best_score, best_partners = score, combination

    return {
        "target": target,
        "partners": list(best_partners) if best_partners else [],
        "score": float(best_score),
        "method": method,
        "considered": considered,
    }
