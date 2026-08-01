r"""Goodness-of-fit testing for copulas.

Two ways to get a p-value out of a Cramer-von Mises statistic, because its null
distribution depends on the unknown parameter and so cannot be tabulated.

**Parametric bootstrap** (``simulation="pb"``). Simulate from the fitted copula,
refit, recompute the statistic, repeat. Correct and simple, and the reference
procedure of Genest, Remillard & Beaudoin (2009) -- but it refits the copula on
every replicate, so a thousand replicates cost a thousand fits.

**Multiplier bootstrap** (``simulation="mult"``). Replicate the limiting
empirical process directly instead of resampling. Because
:math:`\sqrt n(\hat\theta - \theta) \approx H^{-1} n^{-1/2}\sum_i W_i` for the
same influence function :math:`W_i` the standard errors already use, a
bootstrap replicate is obtained by reweighting with random multipliers
:math:`Z_i` -- **no refitting at all**. Orders of magnitude faster, and the
procedure that makes goodness-of-fit practical at large ``n``.

The multiplier bootstrap has no implementation anywhere else in Python.

The p-value follows Pesarin's convention,
:math:`(0.5 + \#\{T_b \ge T\})/(N+1)`, which is strictly inside ``(0, 1)`` --
a p-value of exactly zero would be an artefact of finite ``N``, not evidence.

References
----------
Genest, C., Remillard, B. and Beaudoin, D. (2009). Goodness-of-fit tests for
    copulas: a review and a power study.
    *Insurance: Mathematics and Economics* 44(2), 199-213.
Kojadinovic, I., Yan, J. and Holmes, M. (2011). Fast large-sample goodness-of-fit
    tests for copulas. *Statistica Sinica* 21(2), 841-871.
    The multiplier bootstrap implemented here.
Remillard, B. and Scaillet, O. (2009). Testing for equality between two copulas.
    *Journal of Multivariate Analysis* 100(3), 377-386.
Pesarin, F. (2001). *Multivariate Permutation Tests*. Wiley.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rcopula.core.base import Copula
from rcopula.dependence import pseudo_obs
from rcopula.fit import fit
from rcopula.fit.variance import mpl_influence
from rcopula.gof.statistics import empirical_copula_at, gof_statistic

__all__ = ["GofResult", "gof_test"]


class GofResult(NamedTuple):
    """Outcome of a goodness-of-fit test.

    Follows the ``scipy.stats`` convention of exposing ``statistic`` and
    ``pvalue``, so it unpacks like any other test result.
    """

    statistic: float
    pvalue: float
    method: str
    simulation: str
    n_rep: int
    copula: Copula

    def __repr__(self) -> str:
        return (
            f"GofResult(statistic={self.statistic:.6g}, pvalue={self.pvalue:.4g}, "
            f"method={self.method!r}, simulation={self.simulation!r})"
        )


def _pesarin_pvalue(replicates: NDArray[np.float64], observed: float) -> float:
    """``(0.5 + #{T_b >= T}) / (N + 1)`` -- never exactly 0 or 1."""
    return float((0.5 + np.sum(replicates >= observed)) / (replicates.size + 1))


def _empirical_partials(u: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Partial derivatives of :math:`C_n` at the data points.

    Central differences with bandwidth :math:`n^{-1/2}`, as in Remillard &
    Scaillet (2009). A step function has no true derivative, so some smoothing
    is unavoidable; this is the standard choice.
    """
    n, d = u.shape
    b = 1.0 / np.sqrt(n)
    out = np.empty((n, d))
    for k in range(d):
        hi, lo = u.copy(), u.copy()
        hi[:, k] = np.minimum(u[:, k] + b, 1.0)
        lo[:, k] = np.maximum(u[:, k] - b, 0.0)
        width = hi[:, k] - lo[:, k]
        out[:, k] = (empirical_copula_at(u, hi) - empirical_copula_at(u, lo)) / width
    return out


def _cdf_gradient(
    copula: Copula, u: NDArray[np.float64], theta: NDArray[np.float64]
) -> NDArray[np.float64]:
    r""":math:`\partial C_\theta(u)/\partial\theta` at each data point."""
    cols = []
    for j in range(theta.size):
        h = 1e-5 * max(abs(theta[j]), 1.0)
        hi, lo = theta.copy(), theta.copy()
        hi[j] += h
        lo[j] -= h
        cols.append((copula.with_params(hi).cdf(u) - copula.with_params(lo).cdf(u)) / (2 * h))
    return np.column_stack(cols)


def _parametric_bootstrap(
    template: Copula,
    fitted: Copula,
    u: NDArray[np.float64],
    method: str,
    estim_method: str,
    n_rep: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Simulate, refit and recompute -- the reference procedure."""
    n = u.shape[0]
    out = np.empty(n_rep)
    for b in range(n_rep):
        sample = pseudo_obs(fitted.rvs(n, random_state=rng))
        refit = fit(template, sample, method=estim_method, estimate_variance=False)
        out[b] = gof_statistic(np.asarray(sample), refit.copula, method=method)
    return out


def _multiplier_bootstrap(
    template: Copula,
    fitted: Copula,
    u: NDArray[np.float64],
    theta: NDArray[np.float64],
    n_rep: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    r"""Replicate the empirical process without refitting.

    Each replicate of :math:`\sqrt n (C_n - C_{\hat\theta})` is

    .. math::

        \hat{\mathbb{C}}^{(b)}(u) = \hat\alpha^{(b)}(u)
            - \sum_k \frac{\partial C_n}{\partial u_k}(u)\,\hat\alpha_k^{(b)}(u_k)
            - \nabla_\theta C_\theta(u)^{\top}\,\hat\Theta^{(b)},

    with :math:`\hat\alpha^{(b)}` the multiplier version of the empirical
    process, the middle term correcting for the estimated margins, and
    :math:`\hat\Theta^{(b)} = H^{-1} n^{-1/2}\sum_i Z_i W_i` the replicate of
    the parameter estimation error.
    """
    n = u.shape[0]
    root_n = np.sqrt(n)

    cn = empirical_copula_at(u)
    partials = _empirical_partials(u)
    grad = _cdf_gradient(template, u, theta)

    influence, hessian = mpl_influence(lambda uu, t: template.with_params(t).logpdf(uu), u, theta)
    h_inv = np.linalg.inv(hessian)

    # indicator[j, i] = 1 if observation i lies below evaluation point j.
    indicator = np.all(u[None, :, :] <= u[:, None, :], axis=2).astype(np.float64)
    per_margin = [
        (u[:, k][None, :] <= u[:, k][:, None]).astype(np.float64) for k in range(u.shape[1])
    ]

    out = np.empty(n_rep)
    for b in range(n_rep):
        # np.asarray only to satisfy the type checker: the overload for a
        # non-None `size` is not narrowed automatically.
        z = np.asarray(rng.normal(size=n), dtype=np.float64)
        z -= z.mean()  # centring is what makes the replicate mean-zero

        alpha = (indicator @ z - cn * z.sum()) / root_n
        for k, ind_k in enumerate(per_margin):
            alpha -= partials[:, k] * ((ind_k @ z) / root_n)

        theta_rep = h_inv @ (influence.T @ z) / root_n
        process = alpha - grad @ theta_rep
        out[b] = float(np.sum(process**2) / n)
    return out


def gof_test(
    copula: Copula,
    data: ArrayLike,
    *,
    method: str = "Sn",
    simulation: str = "pb",
    estim_method: str = "mpl",
    n_rep: int = 1000,
    random_state: np.random.Generator | int | None = None,
    ties_method: str = "average",
) -> GofResult:
    """Test whether a copula family fits the data (R's ``gofCopula``).

    Parameters
    ----------
    copula : Copula
        The family to test. Parameters are estimated from the data.
    data : array_like
        ``(n, d)`` observations. Always rank-transformed, whatever the scale.
    method : {"Sn", "Tn", "AnChisq", "AnGamma"}
        Which statistic to use. ``Sn`` is the default and the recommended one.
    simulation : {"pb", "mult"}
        ``"pb"`` refits on every replicate; ``"mult"`` reweights instead and is
        far faster. ``"mult"`` supports ``Sn`` only.
    estim_method : str
        Estimation method passed to :func:`~rcopula.fit`.
    n_rep : int
        Number of bootstrap replicates.
    random_state : Generator, int or None
        Seed or generator.

    Returns
    -------
    GofResult
        With ``statistic`` and ``pvalue``, as in ``scipy.stats``.

    Notes
    -----
    A large p-value is *not* evidence that the family is correct -- only that
    this test could not reject it at this sample size.

    Examples
    --------
    The right family is not rejected; the wrong one is:

    >>> import numpy as np
    >>> from rcopula import ClaytonCopula, GumbelCopula, gof_test
    >>> x = ClaytonCopula(4.0).rvs(300, random_state=0)
    >>> right = gof_test(ClaytonCopula(), x, n_rep=100, random_state=1)
    >>> wrong = gof_test(GumbelCopula(), x, n_rep=100, random_state=1)
    >>> bool(right.pvalue > 0.05)
    True
    >>> bool(wrong.pvalue < 0.05)
    True

    The multiplier bootstrap reaches the same conclusion without refitting:

    >>> fast = gof_test(GumbelCopula(), x, simulation="mult", n_rep=100, random_state=1)
    >>> bool(fast.pvalue < 0.05)
    True
    """
    if simulation not in ("pb", "mult"):
        raise ValueError(f"simulation must be 'pb' or 'mult', got {simulation!r}")
    if simulation == "mult" and method != "Sn":
        raise ValueError(
            "the multiplier bootstrap is implemented for method='Sn' only; "
            f"got method={method!r}. Use simulation='pb' for the others."
        )

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    # Always rank-transform, unlike `fit`, which passes unit-cube input through
    # unchanged. The Cramer-von Mises statistic compares the *empirical* copula
    # against the fitted one, and the empirical copula is defined on ranks; its
    # asymptotic null distribution assumes as much. Feeding exact copula draws
    # through untransformed inflates Sn by an order of magnitude and rejects
    # correct models. R's `gofCopula` applies `pobs` internally for the same
    # reason.
    u = np.asarray(pseudo_obs(np.asarray(data, dtype=np.float64), ties_method=ties_method))

    result = fit(copula, u, method=estim_method, estimate_variance=False)
    observed = gof_statistic(u, result.copula, method=method)

    if simulation == "pb":
        replicates = _parametric_bootstrap(
            copula, result.copula, u, method, estim_method, n_rep, rng
        )
    else:
        replicates = _multiplier_bootstrap(copula, result.copula, u, result.params, n_rep, rng)

    return GofResult(
        statistic=observed,
        pvalue=_pesarin_pvalue(replicates, observed),
        method=method,
        simulation=simulation,
        n_rep=n_rep,
        copula=result.copula,
    )
