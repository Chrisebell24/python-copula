r"""Stirling and Eulerian numbers.

These are the coefficient machinery behind the ``d``-dimensional Archimedean
densities. The Gumbel generator's ``d``-th derivative, in particular, is a
polynomial whose coefficients are built from Stirling numbers of *both* kinds:

.. math::

    a_{d,k}(\theta) = (-1)^{d-k} \sum_{j=k}^{d}
        \theta^{-j}\, s(d, j)\, S(j, k)

as given in

    Hofert, M., Mächler, M. and McNeil, A. J. (2012). Likelihood inference for
    Archimedean copulas in high dimensions under known margins.
    *Journal of Multivariate Analysis* 110, 133-150.

Exact integer recurrences are used and results are cached, because these are
evaluated inside likelihood loops and the values grow large enough that
recomputing in floating point would both cost time and lose digits.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "eulerian",
    "eulerian_all",
    "stirling1",
    "stirling1_all",
    "stirling2",
    "stirling2_all",
]


@lru_cache(maxsize=256)
def _stirling1_row(n: int) -> tuple[int, ...]:
    """Signed Stirling numbers of the first kind ``s(n, k)`` for ``k = 0..n``.

    Recurrence ``s(n+1, k) = s(n, k-1) - n * s(n, k)``.
    """
    row: tuple[int, ...] = (1,)
    for m in range(n):
        prev = (0, *row, 0)
        row = tuple(prev[k - 1 + 1] - m * prev[k + 1] for k in range(m + 2))
    return row


@lru_cache(maxsize=256)
def _stirling2_row(n: int) -> tuple[int, ...]:
    """Stirling numbers of the second kind ``S(n, k)`` for ``k = 0..n``.

    Recurrence ``S(n+1, k) = k * S(n, k) + S(n, k-1)``.
    """
    row: tuple[int, ...] = (1,)
    for m in range(n):
        prev = (0, *row, 0)
        row = tuple(k * prev[k + 1] + prev[k] for k in range(m + 2))
    return row


@lru_cache(maxsize=256)
def _eulerian_row(n: int) -> tuple[int, ...]:
    """Eulerian numbers ``A(n, k)`` for ``k = 0..n``.

    Recurrence ``A(n, k) = (k + 1) * A(n-1, k) + (n - k) * A(n-1, k-1)``.
    """
    row: tuple[int, ...] = (1,)
    for m in range(1, n + 1):
        prev = (0, *row, 0)
        row = tuple((k + 1) * prev[k + 1] + (m - k) * prev[k] for k in range(m + 1))
    return row


def stirling1(n: int, k: int) -> float:
    """Return the signed Stirling number of the first kind, ``s(n, k)``.

    A whole number from combinatorics. In plain terms, ``|s(n, k)|`` counts the
    ways to arrange ``n`` labelled objects into ``k`` cycles; the sign is
    ``(-1)**(n - k)``. Together with :func:`stirling2` it gives the polynomial
    coefficients of the Gumbel generator's high-order derivatives, which the
    ``d``-dimensional Gumbel density needs. You will rarely call it directly.

    Parameters
    ----------
    n : int
        Number of objects, ``n >= 0``.
    k : int
        Number of cycles, ``k >= 0``.

    Returns
    -------
    float
        ``s(n, k)``, computed exactly in integer arithmetic and returned as a
        float. ``0.0`` when ``k > n``.

    Raises
    ------
    ValueError
        If ``n`` or ``k`` is negative.

    Examples
    --------
    >>> from rcopula.special.combinatorics import stirling1
    >>> [stirling1(4, k) for k in range(5)]
    [0.0, -6.0, 11.0, -6.0, 1.0]
    """
    if n < 0 or k < 0:
        raise ValueError(f"stirling1 requires non-negative n, k; got n={n}, k={k}")
    if k > n:
        return 0.0
    return float(_stirling1_row(n)[k])


def stirling1_all(n: int) -> NDArray[np.float64]:
    """Return a whole row of signed Stirling numbers of the first kind at once.

    Equivalent to ``[stirling1(n, k) for k in range(1, n + 1)]`` but taken from
    the same cached row in one go.

    Parameters
    ----------
    n : int
        Number of objects, ``n >= 0``.

    Returns
    -------
    numpy.ndarray of float64, shape (n,)
        ``s(n, k)`` for ``k = 1, ..., n``. Empty when ``n = 0``.

    Examples
    --------
    >>> from rcopula.special.combinatorics import stirling1_all
    >>> stirling1_all(4).tolist()
    [-6.0, 11.0, -6.0, 1.0]
    """
    return np.asarray(_stirling1_row(n)[1:], dtype=np.float64)


def stirling2(n: int, k: int) -> float:
    """Count the ways to split ``n`` labelled items into ``k`` non-empty groups.

    This is the Stirling number of the second kind ``S(n, k)``: the number of
    ways to partition ``n`` labelled objects into ``k`` non-empty unlabelled
    subsets. Together with :func:`stirling1` it gives the polynomial
    coefficients of the Gumbel generator's high-order derivatives, which the
    ``d``-dimensional Gumbel density needs.

    Parameters
    ----------
    n : int
        Number of objects, ``n >= 0``.
    k : int
        Number of groups, ``k >= 0``.

    Returns
    -------
    float
        ``S(n, k)``, computed exactly in integer arithmetic and returned as a
        float. ``0.0`` when ``k > n``.

    Raises
    ------
    ValueError
        If ``n`` or ``k`` is negative.

    Examples
    --------
    >>> from rcopula.special.combinatorics import stirling2
    >>> [stirling2(4, k) for k in range(5)]
    [0.0, 1.0, 7.0, 6.0, 1.0]
    """
    if n < 0 or k < 0:
        raise ValueError(f"stirling2 requires non-negative n, k; got n={n}, k={k}")
    if k > n:
        return 0.0
    return float(_stirling2_row(n)[k])


def stirling2_all(n: int) -> NDArray[np.float64]:
    """Return a whole row of Stirling numbers of the second kind at once.

    Equivalent to ``[stirling2(n, k) for k in range(1, n + 1)]`` but taken from
    the same cached row in one go.

    Parameters
    ----------
    n : int
        Number of objects, ``n >= 0``.

    Returns
    -------
    numpy.ndarray of float64, shape (n,)
        ``S(n, k)`` for ``k = 1, ..., n``. Empty when ``n = 0``.

    Examples
    --------
    >>> from rcopula.special.combinatorics import stirling2_all
    >>> stirling2_all(4).tolist()
    [1.0, 7.0, 6.0, 1.0]
    """
    return np.asarray(_stirling2_row(n)[1:], dtype=np.float64)


def eulerian(n: int, k: int) -> float:
    """Count the orderings of ``n`` items that go up exactly ``k`` times.

    This is the Eulerian number ``A(n, k)``: the number of permutations of
    ``1..n`` with exactly ``k`` ascents (positions where the next entry is
    larger). The package uses them
    as the coefficients of the polylogarithm of negative integer order, which
    the ``d``-dimensional densities of several Archimedean families are built
    from.

    Parameters
    ----------
    n : int
        Permutation length, ``n >= 0``.
    k : int
        Number of ascents, ``k >= 0``.

    Returns
    -------
    float
        ``A(n, k)``, computed exactly in integer arithmetic and returned as a
        float. ``0.0`` when ``k > n``; note that ``A(n, n) = 0`` for ``n >= 1``.

    Raises
    ------
    ValueError
        If ``n`` or ``k`` is negative.

    Examples
    --------
    >>> from rcopula.special.combinatorics import eulerian
    >>> [eulerian(4, k) for k in range(4)]
    [1.0, 11.0, 11.0, 1.0]
    """
    if n < 0 or k < 0:
        raise ValueError(f"eulerian requires non-negative n, k; got n={n}, k={k}")
    if k > n:
        return 0.0
    return float(_eulerian_row(n)[k])


def eulerian_all(n: int) -> NDArray[np.float64]:
    """Return a whole row of Eulerian numbers at once.

    Equivalent to ``[eulerian(n, k) for k in range(n)]`` (the entries that can
    be non-zero) but taken from the same cached row in one go.

    Parameters
    ----------
    n : int
        Permutation length, ``n >= 0``.

    Returns
    -------
    numpy.ndarray of float64, shape (max(n, 1),)
        ``A(n, k)`` for ``k = 0, ..., n - 1``. For ``n = 0`` this is ``[1.0]``.

    Examples
    --------
    >>> from rcopula.special.combinatorics import eulerian_all
    >>> eulerian_all(4).tolist()
    [1.0, 11.0, 11.0, 1.0]
    """
    return np.asarray(_eulerian_row(n)[: max(n, 1)], dtype=np.float64)
