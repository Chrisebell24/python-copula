"""Numerical special functions underpinning the copula families.

These are the pieces that are hardest to get right and that have no
ready-made implementation in ``scipy``: accurate ``log(1 - exp(-a))`` and
``log(1 + exp(x))``, the Debye functions (needed for Frank's Kendall tau and
Spearman's rho), the combinatorial coefficient machinery for Archimedean
densities, stable-law sampling, and a reproducible multivariate normal/t CDF.

Every module here cites the paper it was implemented from.
"""

from __future__ import annotations

from rcopula.special.debye import debye1, debye2, debye_n
from rcopula.special.logexp import log1mexp, log1pexp, signed_logsumexp

__all__ = [
    "debye1",
    "debye2",
    "debye_n",
    "log1mexp",
    "log1pexp",
    "signed_logsumexp",
]
