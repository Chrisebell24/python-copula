"""Tests for the copula-GARCH model.

There is no R ``copula`` oracle for this -- R's copula-GARCH vignette delegates
the marginal models to ``rugarch``. The margins are checked against ``rugarch``
itself (fixtures from ``tools/rgolden/10_garch.R``: GARCH, GJR-GARCH, an
ARMA(1,1) mean and Student-t innovations); everything else is validated against
properties that hold exactly:

* **Parameter recovery** from series simulated with known GARCH parameters.
* **Scale equivariance**: GARCH is exactly equivariant under ``x -> c*x``, which
  pins the internal rescaling used to keep the optimiser well conditioned.
* **The recursion itself**, against a literal Python loop.
* **Filtering removes volatility clustering** -- the reason for the first step.
* **The copula is recovered in the innovations** even when the raw returns share
  a volatility regime that makes them look dependent when they are not.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import signal, stats

import rcopula as rc
from rcopula.garch import (
    _MAX_PERSISTENCE,
    CopulaGarch,
    GarchResult,
    _filter_variance,
    _fit_reparameterised,
    _Layout,
    _mean_residuals,
    _neg_loglik,
    fit_garch,
)


def simulate_garch(
    n: int,
    omega: float = 0.05,
    alpha: float = 0.10,
    beta: float = 0.85,
    mu: float = 0.0,
    df: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """A GARCH(1,1) series, written as the definition rather than the filter."""
    rng = np.random.default_rng(seed)
    z = (
        rng.standard_normal(n)
        if df is None
        else stats.t(df=df, scale=np.sqrt((df - 2) / df)).rvs(n, random_state=rng)
    )
    x = np.empty(n)
    s2, e = omega / (1.0 - alpha - beta), 0.0
    for i in range(n):
        s2 = omega + alpha * e**2 + beta * s2
        e = np.sqrt(s2) * z[i]
        x[i] = mu + e
    return x


class TestVarianceFilter:
    def test_matches_a_literal_loop(self) -> None:
        """The lfilter shortcut must be the recursion, not merely close to it."""
        rng = np.random.default_rng(0)
        eps = rng.standard_normal(500)
        omega, alpha, beta, s0 = 0.04, 0.12, 0.83, 1.3

        expected = np.empty(500)
        expected[0] = s0
        for t in range(1, 500):
            expected[t] = omega + alpha * eps[t - 1] ** 2 + beta * expected[t - 1]

        assert np.allclose(_filter_variance(eps, omega, alpha, beta, s0), expected, rtol=1e-12)

    def test_zero_alpha_decays_geometrically(self) -> None:
        eps = np.zeros(50)
        v = _filter_variance(eps, 0.0, 0.0, 0.5, 4.0)
        assert np.allclose(v, 4.0 * 0.5 ** np.arange(50))

    def test_lfilter_is_the_bottleneck_free_path(self) -> None:
        """Guards the signature we rely on: same answer via scipy directly."""
        eps = np.random.default_rng(1).standard_normal(100)
        drive = 0.05 + 0.1 * eps[:-1] ** 2
        tail = signal.lfilter([1.0], [1.0, -0.85], drive, zi=np.array([0.85]))[0]
        assert np.allclose(_filter_variance(eps, 0.05, 0.1, 0.85, 1.0)[1:], tail)


class TestFitGarch:
    def test_recovers_known_parameters(self) -> None:
        x = simulate_garch(12_000, omega=0.05, alpha=0.10, beta=0.85, seed=3)
        res = fit_garch(x)
        assert res.alpha == pytest.approx(0.10, abs=0.03)
        assert res.beta == pytest.approx(0.85, abs=0.04)
        assert res.unconditional_vol == pytest.approx(1.0, rel=0.15)

    def test_recovers_the_mean(self) -> None:
        x = simulate_garch(8000, mu=0.5, seed=4)
        assert fit_garch(x).mu == pytest.approx(0.5, abs=0.05)

    def test_recovers_the_degrees_of_freedom(self) -> None:
        x = simulate_garch(12_000, df=5.0, seed=5)
        res = fit_garch(x, dist="t")
        assert res.df is not None
        assert res.df == pytest.approx(5.0, rel=0.25)

    def test_student_t_fits_fat_tails_better(self) -> None:
        x = simulate_garch(6000, df=4.0, seed=6)
        assert fit_garch(x, dist="t").loglik > fit_garch(x, dist="normal").loglik

    def test_quasi_mle_still_recovers_the_variance_parameters(self) -> None:
        """The Bollerslev-Wooldridge result: normal QMLE is consistent anyway."""
        x = simulate_garch(12_000, alpha=0.10, beta=0.85, df=4.0, seed=7)
        res = fit_garch(x, dist="normal")
        assert res.alpha == pytest.approx(0.10, abs=0.04)
        assert res.beta == pytest.approx(0.85, abs=0.06)

    @pytest.mark.parametrize("scale", [2.0**-10, 2.0**10])
    def test_is_exactly_scale_equivariant(self, scale: float) -> None:
        """x -> c*x sends (mu, omega) -> (c*mu, c^2*omega) and fixes alpha, beta.

        The optimiser sees a unit-variance series either way, so this is a test
        that the rescaling is undone correctly -- not a statistical property.
        Powers of two make it hold *bitwise*: scaling every value by a power of
        two shifts exponents only, so ``std(c*x)`` is exactly ``c*std(x)`` and
        the optimiser sees a byte-identical series. Other factors agree only to
        the optimiser's tolerance -- see the test below.
        """
        x = simulate_garch(3000, mu=0.2, seed=8)
        a, b = fit_garch(x), fit_garch(scale * x)
        assert b.alpha == pytest.approx(a.alpha, rel=1e-10)
        assert b.beta == pytest.approx(a.beta, rel=1e-10)
        assert b.mu == pytest.approx(scale * a.mu, rel=1e-10)
        assert b.omega == pytest.approx(scale**2 * a.omega, rel=1e-10)
        assert b.sigma[-1] == pytest.approx(scale * a.sigma[-1], rel=1e-10)
        assert np.allclose(b.resid, a.resid, rtol=1e-10)
        # Change of variables: log f_{cX}(y) = log f_X(y/c) - log c.
        assert b.loglik == pytest.approx(a.loglik - x.size * np.log(scale), rel=1e-10)

    @pytest.mark.parametrize("scale", [1e-3, 100.0])
    def test_is_scale_equivariant_for_arbitrary_factors(self, scale: float) -> None:
        """Same identity for factors that are not powers of two.

        Here ``std(c*x)`` differs from ``c*std(x)`` in the last bits, so the two
        optimiser runs start from marginally different series and agree to their
        convergence tolerance rather than exactly.
        """
        x = simulate_garch(3000, mu=0.2, seed=8)
        a, b = fit_garch(x), fit_garch(scale * x)
        assert b.alpha == pytest.approx(a.alpha, rel=1e-5)
        assert b.beta == pytest.approx(a.beta, rel=1e-5)
        assert b.mu == pytest.approx(scale * a.mu, rel=1e-5)
        assert b.omega == pytest.approx(scale**2 * a.omega, rel=1e-5)

    def test_filtering_removes_volatility_clustering(self) -> None:
        x = simulate_garch(6000, alpha=0.12, beta=0.85, seed=9)
        res = fit_garch(x)
        raw = np.corrcoef(x[1:] ** 2, x[:-1] ** 2)[0, 1]
        filtered = np.corrcoef(res.resid[1:] ** 2, res.resid[:-1] ** 2)[0, 1]
        assert raw > 0.15
        assert abs(filtered) < 0.25 * raw

    def test_residuals_are_standardised(self) -> None:
        res = fit_garch(simulate_garch(6000, seed=10))
        assert res.resid.std() == pytest.approx(1.0, abs=0.05)
        assert res.resid.mean() == pytest.approx(0.0, abs=0.05)

    def test_residuals_reproduce_the_series(self) -> None:
        x = simulate_garch(2000, mu=0.3, seed=11)
        res = fit_garch(x)
        assert np.allclose(res.mu + res.sigma * res.resid, x, rtol=1e-10)

    def test_diagnostics_are_coherent(self) -> None:
        res = fit_garch(simulate_garch(4000, seed=12))
        assert 0.0 < res.persistence < 1.0
        assert res.half_life > 0.0
        assert res.aic == pytest.approx(2 * res.n_params - 2 * res.loglik)
        assert res.bic > res.aic  # log(n) > 2 for n >= 8
        assert res.n_params == 4

    def test_t_innovation_has_unit_variance(self) -> None:
        res = fit_garch(simulate_garch(3000, df=6.0, seed=13), dist="t")
        assert res.innovation().var() == pytest.approx(1.0, rel=1e-10)
        assert res.n_params == 5

    def test_repr_names_the_key_numbers(self) -> None:
        res = fit_garch(simulate_garch(1000, seed=14), name="SPX")
        assert "SPX" in repr(res)
        assert "persistence" in repr(res)

    def test_rejects_unusable_input(self) -> None:
        with pytest.raises(ValueError, match="at least 50"):
            fit_garch(np.zeros(10))
        with pytest.raises(ValueError, match="constant"):
            fit_garch(np.ones(100))
        with pytest.raises(ValueError, match="non-finite"):
            fit_garch(np.concatenate([np.random.default_rng(0).standard_normal(100), [np.nan]]))
        with pytest.raises(ValueError, match="dist must be"):
            fit_garch(simulate_garch(200, seed=0), dist="ged")


class TestForecastVariance:
    def test_one_step_ahead_matches_the_recursion(self) -> None:
        res = fit_garch(simulate_garch(2000, seed=15))
        eps_last = res.resid[-1] * res.sigma[-1]
        manual = res.omega + res.alpha * eps_last**2 + res.beta * res.sigma[-1] ** 2
        assert res.forecast_variance(1)[0] == pytest.approx(manual, rel=1e-12)

    def test_converges_to_the_unconditional_level(self) -> None:
        res = fit_garch(simulate_garch(3000, seed=16))
        v = res.forecast_variance(1000)
        assert v[-1] == pytest.approx(res.unconditional_vol**2, rel=1e-3)

    def test_is_monotone_towards_the_long_run(self) -> None:
        res = fit_garch(simulate_garch(3000, seed=17))
        v = res.forecast_variance(200)
        long_run = res.unconditional_vol**2
        gaps = np.abs(v - long_run)
        assert np.all(np.diff(gaps) <= 1e-15)

    def test_vol_is_the_square_root(self) -> None:
        res = fit_garch(simulate_garch(1000, seed=18))
        assert np.allclose(res.forecast_vol(20) ** 2, res.forecast_variance(20))

    def test_rejects_a_zero_horizon(self) -> None:
        with pytest.raises(ValueError, match="horizon must be"):
            fit_garch(simulate_garch(500, seed=19)).forecast_variance(0)


def _common_volatility_returns(seed: int, n: int = 4000) -> np.ndarray:
    """Independent series sharing a smooth AR(1) log-volatility process.

    Smooth rather than regime-switching, so that a GARCH(1,1) can actually track
    it -- a step-function volatility pins the fit against the IGARCH boundary and
    leaves part of the artifact unfiltered.
    """
    rng = np.random.default_rng(seed)
    h, e = np.zeros(n), rng.standard_normal(n)
    for t in range(1, n):
        h[t] = 0.98 * h[t - 1] + 0.25 * e[t]
    return np.asarray(rng.standard_normal((n, 2)) * (0.01 * np.exp(h))[:, None])


class TestCopulaGarchFit:
    def test_common_volatility_is_not_mistaken_for_tail_dependence(self) -> None:
        """The headline reason to filter first.

        Two genuinely independent series driven by a common volatility process
        look strongly **tail dependent**: in a high-volatility stretch both are
        large at once, purely because volatility is shared. A t copula fitted to
        the raw returns reports df near 1; fitted to the GARCH innovations it
        reports several times that, with correspondingly little tail dependence.

        Note the artifact does *not* show up in rank correlation, which is why
        it survives casual inspection -- see the companion assertion below.
        """
        r = _common_volatility_returns(seed=1)
        naive = rc.fit(rc.StudentCopula(0.0, df=8.0), rc.pseudo_obs(r), method="mpl")
        model = CopulaGarch.fit(r, rc.StudentCopula(0.0, df=8.0, dim=2))

        assert naive.copula.df < 2.0
        assert model.copula.df > 3.5
        assert naive.copula.lambda_().upper > 3.0 * model.copula.lambda_().upper
        assert abs(naive.copula.params[0]) < 0.06
        assert abs(model.copula.params[0]) < 0.06

    def test_the_squared_returns_are_where_the_artifact_lives(self) -> None:
        """Confirms the mechanism: co-moving magnitudes, not co-moving signs."""
        r = _common_volatility_returns(seed=1)
        model = CopulaGarch.fit(r, rc.GaussianCopula(0.0, dim=2))
        z = np.column_stack([m.resid for m in model.margins])

        # Rank correlation of the magnitudes: Pearson on squared heavy-tailed
        # returns is dominated by a handful of observations and is far noisier.
        raw = stats.spearmanr(r[:, 0] ** 2, r[:, 1] ** 2).statistic
        filtered = stats.spearmanr(z[:, 0] ** 2, z[:, 1] ** 2).statistic
        assert raw > 0.3
        assert abs(filtered) < 0.3 * raw

    def test_recovers_an_injected_copula(self) -> None:
        """Dependence put into the innovations comes back out of the fit."""
        rng = np.random.default_rng(2)
        u = rc.ClaytonCopula.from_tau(0.5).rvs(4000, random_state=rng)
        z = stats.norm.ppf(u)
        r = np.column_stack([_apply_garch(z[:, j], 0.05, 0.1, 0.85) for j in range(2)])
        model = CopulaGarch.fit(r, rc.ClaytonCopula(1.0))
        assert model.copula.tau() == pytest.approx(0.5, abs=0.05)

    def test_keeps_dataframe_column_names(self) -> None:
        rng = np.random.default_rng(3)
        frame = pd.DataFrame(rng.standard_normal((800, 2)) * 0.01, columns=["SPX", "UST"])
        model = CopulaGarch.fit(frame, rc.GaussianCopula(0.0, dim=2))
        assert model.names == ["SPX", "UST"]
        assert list(model.summary().index) == ["SPX", "UST"]

    def test_summary_has_a_row_per_margin(self) -> None:
        rng = np.random.default_rng(4)
        model = CopulaGarch.fit(rng.standard_normal((800, 3)) * 0.01, rc.GaussianCopula(0.2, dim=3))
        summary = model.summary()
        assert summary.shape[0] == 3
        assert np.all(summary["persistence"] < 1.0)

    def test_rejects_mismatched_shapes(self) -> None:
        rng = np.random.default_rng(5)
        r = rng.standard_normal((500, 2)) * 0.01
        with pytest.raises(ValueError, match="dim="):
            CopulaGarch.fit(r, rc.GaussianCopula(0.2, dim=3))
        with pytest.raises(ValueError, match="2-d"):
            CopulaGarch.fit(r[:, 0], rc.GaussianCopula(0.2, dim=2))
        with pytest.raises(ValueError, match="dim="):
            CopulaGarch([fit_garch(r[:, 0])], rc.GaussianCopula(0.2, dim=2))
        with pytest.raises(ValueError, match="innovations must be"):
            CopulaGarch(
                [fit_garch(r[:, j]) for j in range(2)],
                rc.GaussianCopula(0.2, dim=2),
                innovations="bootstrap",
            )


def _apply_garch(z: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """Push given innovations through a GARCH recursion."""
    x = np.empty_like(z)
    s2, e = omega / (1.0 - alpha - beta), 0.0
    for i in range(z.size):
        s2 = omega + alpha * e**2 + beta * s2
        e = np.sqrt(s2) * z[i]
        x[i] = e
    return x


def _two_margins(seed: int = 0, n: int = 1500) -> list[GarchResult]:
    rng = np.random.default_rng(seed)
    r = rng.standard_normal((n, 2)) * 0.01
    return [fit_garch(r[:, j], name=f"a{j}") for j in range(2)]


class TestSimulation:
    def test_shape_and_dependence(self) -> None:
        model = CopulaGarch(_two_margins(), rc.ClaytonCopula.from_tau(0.5))
        paths = model.simulate(horizon=3, n=8000, random_state=0)
        assert paths.shape == (8000, 3, 2)
        tau = stats.kendalltau(paths[:, 0, 0], paths[:, 0, 1]).statistic
        assert tau == pytest.approx(0.5, abs=0.03)

    def test_dependence_holds_at_every_step(self) -> None:
        model = CopulaGarch(_two_margins(seed=1), rc.GumbelCopula.from_tau(0.4))
        paths = model.simulate(horizon=4, n=6000, random_state=0)
        for step in range(4):
            tau = stats.kendalltau(paths[:, step, 0], paths[:, step, 1]).statistic
            assert tau == pytest.approx(0.4, abs=0.04)

    def test_volatility_clusters_along_the_path(self) -> None:
        """Simulated paths must show the persistence they were fitted with."""
        x = simulate_garch(4000, alpha=0.12, beta=0.85, seed=20)
        margins = [fit_garch(x), fit_garch(simulate_garch(4000, seed=21))]
        model = CopulaGarch(margins, rc.GaussianCopula(0.3))
        paths = model.simulate(horizon=60, n=2000, random_state=0)

        sq = paths[:, :, 0] ** 2
        acf = np.mean([np.corrcoef(row[1:], row[:-1])[0, 1] for row in sq if row.std() > 0])
        assert acf > 0.05

    def test_the_first_step_starts_from_the_observed_state(self) -> None:
        margins = _two_margins(seed=2)
        model = CopulaGarch(margins, rc.GaussianCopula(0.0))
        paths = model.simulate(horizon=1, n=40_000, random_state=0)
        realised = paths[:, 0, 0].std()
        assert realised == pytest.approx(margins[0].forecast_vol(1)[0], rel=0.05)

    def test_forecast_is_the_summed_path(self) -> None:
        model = CopulaGarch(_two_margins(seed=3), rc.GaussianCopula(0.4))
        assert np.allclose(
            model.forecast(horizon=5, n=500, random_state=7),
            model.simulate(horizon=5, n=500, random_state=7).sum(axis=1),
        )

    def test_horizon_widens_the_distribution(self) -> None:
        model = CopulaGarch(_two_margins(seed=4), rc.GaussianCopula(0.4))
        one = model.forecast(horizon=1, n=20_000, random_state=0).std(axis=0)
        ten = model.forecast(horizon=10, n=20_000, random_state=0).std(axis=0)
        assert np.all(ten > 2.0 * one)

    def test_parametric_innovations_reach_beyond_the_sample(self) -> None:
        """Filtered historical simulation is capped by history; parametric is not."""
        margins = _two_margins(seed=5)
        cop = rc.GaussianCopula(0.3)
        cap = max(abs(margins[0].resid).max(), abs(margins[1].resid).max())

        empirical = CopulaGarch(margins, cop, innovations="empirical")
        parametric = CopulaGarch(margins, cop, innovations="parametric")
        e = empirical.simulate(1, 40_000, random_state=0)[:, 0, :]
        p = parametric.simulate(1, 40_000, random_state=0)[:, 0, :]

        worst_sigma = max(m.forecast_vol(1)[0] for m in margins)
        assert np.abs(e).max() <= cap * worst_sigma * 1.001
        assert np.abs(p).max() > np.abs(e).max()

    def test_rejects_a_zero_horizon(self) -> None:
        with pytest.raises(ValueError, match="horizon must be"):
            CopulaGarch(_two_margins(), rc.GaussianCopula(0.3)).simulate(horizon=0)


class TestForecastRisk:
    def test_expected_shortfall_exceeds_var(self) -> None:
        model = CopulaGarch(_two_margins(), rc.StudentCopula(0.6, df=4.0))
        r = model.forecast_risk(alpha=0.99, n=40_000, random_state=0)
        assert r["expected_shortfall"] > r["var"] > 0.0

    def test_var_increases_with_confidence(self) -> None:
        model = CopulaGarch(_two_margins(seed=6), rc.GaussianCopula(0.5))
        levels = [0.90, 0.95, 0.99, 0.995]
        vars_ = [model.forecast_risk(alpha=a, n=60_000, random_state=0)["var"] for a in levels]
        assert all(a < b for a, b in pairwise(vars_))

    def test_tail_dependence_costs_more_at_equal_tau(self) -> None:
        """Same rank correlation, different tail -- and a different capital number."""
        margins = _two_margins(seed=7)
        tau = 0.5
        gauss = CopulaGarch(margins, rc.GaussianCopula.from_tau(tau))
        student = CopulaGarch(margins, rc.StudentCopula.from_tau(tau, df=3.0))
        a = gauss.forecast_risk(alpha=0.995, n=60_000, random_state=0)
        b = student.forecast_risk(alpha=0.995, n=60_000, random_state=0)
        assert b["var"] > a["var"]
        assert b["expected_shortfall"] > a["expected_shortfall"]

    def test_concentration_is_riskier_than_diversifying(self) -> None:
        model = CopulaGarch(_two_margins(seed=8), rc.GaussianCopula(0.2))
        even = model.forecast_risk([0.5, 0.5], alpha=0.99, n=40_000, random_state=0)
        all_in = model.forecast_risk([1.0, 0.0], alpha=0.99, n=40_000, random_state=0)
        assert all_in["var"] > even["var"]

    def test_reports_the_horizon_moments(self) -> None:
        model = CopulaGarch(_two_margins(seed=9), rc.GaussianCopula(0.5))
        r = model.forecast_risk(horizon=5, n=20_000, random_state=0)
        assert r["volatility"] > 0.0
        assert abs(r["mean"]) < 10.0 * r["volatility"]

    def test_rejects_wrong_length_weights(self) -> None:
        model = CopulaGarch(_two_margins(), rc.GaussianCopula(0.3))
        with pytest.raises(ValueError, match="expected 2"):
            model.forecast_risk([0.3, 0.3, 0.4])

    def test_repr_is_informative(self) -> None:
        text = repr(CopulaGarch(_two_margins(), rc.GaussianCopula(0.3)))
        assert "CopulaGarch" in text
        assert "empirical" in text


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_persistence_never_exceeds_one_on_near_integrated_data(seed: int) -> None:
    """Stochastic volatility with persistence 0.98 in log-variance pushes GARCH
    to its boundary. SLSQP can stop there with alpha + beta > 1 (seen on Linux
    CI at 1.0071); the reparameterised refit must keep the model stationary."""
    rng = np.random.default_rng(seed)
    log_vol = np.zeros(4000)
    shocks = rng.standard_normal(4000)
    for t in range(1, 4000):
        log_vol[t] = 0.98 * log_vol[t - 1] + 0.25 * shocks[t]
    x = rng.standard_normal(4000) * 0.01 * np.exp(log_vol)
    for dist in ("normal", "t"):
        assert fit_garch(x, dist=dist).persistence < 1.0


def test_the_reparameterised_refit_cannot_break_stationarity() -> None:
    rng = np.random.default_rng(0)
    y = rng.standard_t(5, size=2000)
    y /= y.std()
    bounds = [(-10.0, 10.0), (1e-8, 10.0), (0.0, _MAX_PERSISTENCE), (0.0, _MAX_PERSISTENCE)]
    theta = _fit_reparameterised(y, "normal", np.array([0.0, 0.05, 0.3, 0.8]), bounds)
    assert theta[2] >= 0.0 and theta[3] >= 0.0
    assert theta[2] + theta[3] <= _MAX_PERSISTENCE + 1e-12


class TestGarchRegressions:
    def test_half_life_without_persistence_is_zero_and_silent(self) -> None:
        """alpha + beta == 0 used to emit a divide-by-zero RuntimeWarning."""
        import dataclasses

        res = _two_margins()[0]
        none = dataclasses.replace(res, alpha=0.0, beta=0.0)
        assert none.half_life == 0.0
        unit = dataclasses.replace(res, alpha=0.1, beta=0.9)
        assert unit.half_life == float("inf")
        half = dataclasses.replace(res, alpha=0.2, beta=0.3)
        assert half.half_life == pytest.approx(1.0)

    def test_simulate_rejects_non_positive_n(self) -> None:
        model = CopulaGarch(_two_margins(), rc.GaussianCopula(0.3))
        with pytest.raises(ValueError, match="n must be"):
            model.simulate(horizon=1, n=0)

    def test_forecast_risk_validates_inputs(self) -> None:
        model = CopulaGarch(_two_margins(), rc.GaussianCopula(0.3))
        with pytest.raises(ValueError, match="horizon must be"):
            model.forecast_risk(horizon=0)
        with pytest.raises(ValueError, match="n must be"):
            model.forecast_risk(n=1)

    def test_forecast_risk_volatility_uses_ddof_1_and_raw_weights(self) -> None:
        model = CopulaGarch(_two_margins(seed=9), rc.GaussianCopula(0.5))
        w = np.array([1.5, -0.5])  # a leveraged long-short book: not rescaled
        r = model.forecast_risk(w, horizon=2, n=50, random_state=3)
        port = model.forecast(horizon=2, n=50, random_state=3) @ w
        assert r["volatility"] == pytest.approx(port.std(ddof=1), rel=1e-12)
        assert r["mean"] == pytest.approx(port.mean(), rel=1e-12)


# ---------------------------------------------------------------------------
# GJR-GARCH and ARMA means
# ---------------------------------------------------------------------------


def simulate_model(
    n: int,
    *,
    mu: float = 0.0,
    phi: float = 0.0,
    theta: float = 0.0,
    omega: float = 0.05,
    alpha: float = 0.05,
    beta: float = 0.85,
    gamma: float = 0.0,
    df: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """ARMA(1,1)-GJR-GARCH(1,1), written out as the definition."""
    rng = np.random.default_rng(seed)
    z = (
        rng.standard_normal(n)
        if df is None
        else stats.t(df=df, scale=np.sqrt((df - 2) / df)).rvs(n, random_state=rng)
    )
    s2 = omega / (1.0 - alpha - gamma / 2 - beta)
    e, r = 0.0, mu / (1.0 - phi)
    x = np.empty(n)
    for i in range(n):
        s2 = omega + (alpha + gamma * (e < 0)) * e**2 + beta * s2
        shock = np.sqrt(s2) * z[i]
        r = mu + phi * r + theta * e + shock
        e = shock
        x[i] = r
    return x


class TestGjrFilter:
    def test_matches_a_literal_loop(self) -> None:
        rng = np.random.default_rng(0)
        eps = rng.standard_normal(400)
        omega, alpha, beta, gamma, s0 = 0.04, 0.03, 0.85, 0.15, 1.2
        expected = np.empty(400)
        expected[0] = s0
        for t in range(1, 400):
            arch = alpha + (gamma if eps[t - 1] < 0 else 0.0)
            expected[t] = omega + arch * eps[t - 1] ** 2 + beta * expected[t - 1]
        got = _filter_variance(eps, omega, alpha, beta, s0, gamma)
        assert np.allclose(got, expected, rtol=1e-12)

    def test_zero_gamma_is_plain_garch_bitwise(self) -> None:
        eps = np.random.default_rng(1).standard_normal(300)
        assert np.array_equal(
            _filter_variance(eps, 0.05, 0.1, 0.85, 1.0, 0.0),
            _filter_variance(eps, 0.05, 0.1, 0.85, 1.0),
        )


class TestGjrFit:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_recovers_known_parameters(self, seed: int) -> None:
        """n = 6000: sampling sd is about 0.01 for alpha, 0.02 for gamma and beta."""
        x = simulate_model(6000, alpha=0.03, gamma=0.12, beta=0.87, seed=seed)
        res = fit_garch(x, vol="gjr")
        assert res.vol == "gjr"
        assert res.alpha == pytest.approx(0.03, abs=0.025)
        assert res.gamma == pytest.approx(0.12, abs=0.05)
        assert res.beta == pytest.approx(0.87, abs=0.04)
        assert res.persistence == pytest.approx(0.96, abs=0.02)
        assert res.unconditional_vol == pytest.approx(1.0, rel=0.2)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_recovers_parameters_with_student_t(self, seed: int) -> None:
        x = simulate_model(6000, alpha=0.03, gamma=0.12, beta=0.87, df=6.0, seed=seed)
        res = fit_garch(x, vol="gjr", dist="t")
        assert res.gamma == pytest.approx(0.12, abs=0.05)
        assert res.beta == pytest.approx(0.87, abs=0.04)
        assert res.df is not None
        assert res.df == pytest.approx(6.0, rel=0.25)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_beats_garch_on_asymmetric_data(self, seed: int) -> None:
        x = simulate_model(5000, alpha=0.03, gamma=0.12, beta=0.87, seed=seed)
        plain, gjr = fit_garch(x), fit_garch(x, vol="gjr")
        # GARCH is GJR with gamma = 0, so the GJR likelihood can only be higher.
        assert gjr.loglik >= plain.loglik - 1e-6
        assert gjr.aic < plain.aic - 10.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_finds_no_asymmetry_where_there_is_none(self, seed: int) -> None:
        x = simulate_model(6000, alpha=0.08, gamma=0.0, beta=0.9, seed=seed)
        res = fit_garch(x, vol="gjr")
        assert abs(res.gamma) < 0.04
        assert res.alpha == pytest.approx(0.08, abs=0.03)

    def test_respects_the_constraints(self) -> None:
        """alpha >= 0, alpha + gamma >= 0 and alpha + gamma/2 + beta < 1."""
        for seed in range(4):
            x = simulate_model(3000, alpha=0.0, gamma=0.2, beta=0.89, seed=seed)
            res = fit_garch(x, vol="gjr")
            assert res.alpha >= 0.0
            assert res.alpha + res.gamma >= -1e-9
            assert res.persistence < 1.0

    def test_negative_gamma_is_allowed(self) -> None:
        """Good news raising volatility more (seen in some commodities) is legal."""
        x = -simulate_model(6000, alpha=0.03, gamma=0.12, beta=0.87, seed=4)
        res = fit_garch(x, vol="gjr")
        # Mirroring the series swaps the roles: alpha + gamma is now the small one.
        assert res.gamma < -0.05
        assert res.alpha + res.gamma == pytest.approx(0.03, abs=0.03)

    def test_is_exactly_scale_equivariant(self) -> None:
        x = simulate_model(3000, mu=0.1, alpha=0.03, gamma=0.12, beta=0.87, seed=5)
        a, b = fit_garch(x, vol="gjr"), fit_garch(2.0**-7 * x, vol="gjr")
        assert b.gamma == pytest.approx(a.gamma, rel=1e-10)
        assert b.omega == pytest.approx(2.0**-14 * a.omega, rel=1e-10)
        assert b.loglik == pytest.approx(a.loglik + 7 * x.size * np.log(2.0), rel=1e-10)

    def test_diagnostics_count_gamma(self) -> None:
        res = fit_garch(simulate_model(2000, gamma=0.1, seed=6), vol="gjr")
        assert res.n_params == 5
        assert fit_garch(simulate_model(2000, seed=6), vol="gjr", dist="t").n_params == 6
        assert res.persistence == pytest.approx(res.alpha + res.gamma / 2 + res.beta)
        assert res.unconditional_vol == pytest.approx(
            np.sqrt(res.omega / (1 - res.alpha - res.gamma / 2 - res.beta))
        )
        assert res.half_life == pytest.approx(np.log(0.5) / np.log(res.persistence))
        assert "gamma=" in repr(res)
        assert "gjr" in repr(res)

    def test_plain_garch_results_have_zero_gamma(self) -> None:
        res = fit_garch(simulate_garch(1000, seed=7))
        assert res.gamma == 0.0
        assert res.phi == 0.0
        assert res.theta == 0.0
        assert res.vol == "garch"
        assert res.mean == "constant"
        assert "gamma" not in repr(res)

    def test_the_reparameterised_refit_cannot_break_the_gjr_constraints(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.standard_t(5, size=2000)
        y /= y.std()
        lay = _Layout("gjr", "constant", "normal")
        bounds = [
            (-10.0, 10.0),
            (1e-8, 10.0),
            (0.0, _MAX_PERSISTENCE),
            (0.0, _MAX_PERSISTENCE),
            (-_MAX_PERSISTENCE, 2 * _MAX_PERSISTENCE),
        ]
        # A start that violates both stationarity and alpha + gamma >= 0.
        start = np.array([0.0, 0.05, 0.3, 0.9, -0.5])
        t = _fit_reparameterised(y, "normal", start, bounds, lay)
        alpha, beta, gamma = t[2], t[3], t[4]
        assert alpha >= 0.0 and beta >= 0.0
        assert alpha + gamma >= -1e-12
        assert alpha + gamma / 2 + beta <= _MAX_PERSISTENCE + 1e-12
        # And it is a genuine optimum, not just a feasible point.
        assert _neg_loglik(t, y, "normal", 1.0, lay) < _neg_loglik(
            np.array([0.0, 0.05, 0.05, 0.85, 0.1]), y, "normal", 1.0, lay
        )


class TestGjrForecast:
    def _fit(self) -> GarchResult:
        return fit_garch(simulate_model(3000, alpha=0.03, gamma=0.15, beta=0.85, seed=8), vol="gjr")

    def test_one_step_uses_the_sign_of_the_last_shock(self) -> None:
        import dataclasses

        res = self._fit()
        s2 = res.sigma[-1] ** 2
        down = dataclasses.replace(res, resid=np.append(res.resid[:-1], -2.0))
        up = dataclasses.replace(res, resid=np.append(res.resid[:-1], 2.0))
        e2 = (2.0 * res.sigma[-1]) ** 2
        assert down.forecast_variance(1)[0] == pytest.approx(
            res.omega + (res.alpha + res.gamma) * e2 + res.beta * s2, rel=1e-12
        )
        assert up.forecast_variance(1)[0] == pytest.approx(
            res.omega + res.alpha * e2 + res.beta * s2, rel=1e-12
        )
        assert down.forecast_variance(1)[0] > up.forecast_variance(1)[0]

    def test_multi_step_decays_at_the_gjr_persistence(self) -> None:
        res = self._fit()
        v = res.forecast_variance(50)
        gaps = v - res.unconditional_vol**2
        assert np.allclose(gaps[1:] / gaps[:-1], res.persistence, rtol=1e-9)

    def test_multi_step_matches_monte_carlo(self) -> None:
        """E[1(eps<0) eps^2] = sigma^2 / 2: the simulated variance agrees.

        Parametric normal innovations through the GJR recursion in
        ``simulate`` must reproduce the analytic forecast at every horizon.
        """
        res = self._fit()
        model = CopulaGarch([res, res], rc.IndependenceCopula(2), innovations="parametric")
        paths = model.simulate(horizon=10, n=200_000, random_state=0)[:, :, 0]
        simulated = paths.var(axis=0)
        assert np.allclose(simulated, res.forecast_variance(10), rtol=0.02)


class TestArmaMean:
    def test_residual_filter_matches_a_literal_loop(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(300)
        mu, phi, theta = 0.1, 0.5, -0.3
        expected = np.empty(300)
        expected[0] = x[0] - mu / (1 - phi)
        for t in range(1, 300):
            expected[t] = x[t] - mu - phi * x[t - 1] - theta * expected[t - 1]
        assert np.allclose(_mean_residuals(x, mu, phi, theta), expected, rtol=1e-12)
        assert np.array_equal(_mean_residuals(x, mu, 0.0, 0.0), x - mu)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_recovers_ar1(self, seed: int) -> None:
        """n = 5000: sampling sd of phi is about 0.013."""
        x = simulate_model(5000, mu=0.05, phi=0.4, omega=0.02, alpha=0.08, beta=0.9, seed=seed)
        res = fit_garch(x, mean="ar1")
        assert res.phi == pytest.approx(0.4, abs=0.05)
        assert res.theta == 0.0
        assert res.unconditional_mean == pytest.approx(0.05 / 0.6, abs=0.06)
        assert res.alpha == pytest.approx(0.08, abs=0.03)
        assert res.beta == pytest.approx(0.9, abs=0.03)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_recovers_arma11(self, seed: int) -> None:
        x = simulate_model(
            5000, mu=0.05, phi=0.6, theta=-0.3, omega=0.02, alpha=0.08, beta=0.9, seed=seed
        )
        res = fit_garch(x, mean="arma11")
        assert res.phi == pytest.approx(0.6, abs=0.07)
        assert res.theta == pytest.approx(-0.3, abs=0.08)
        assert res.alpha == pytest.approx(0.08, abs=0.03)
        assert res.beta == pytest.approx(0.9, abs=0.03)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_recovers_arma_with_gjr_and_student_t(self, seed: int) -> None:
        x = simulate_model(
            5000, phi=0.3, theta=0.2, alpha=0.03, gamma=0.1, beta=0.88, df=5.0, seed=seed
        )
        res = fit_garch(x, mean="arma11", vol="gjr", dist="t")
        assert res.phi == pytest.approx(0.3, abs=0.1)
        assert res.theta == pytest.approx(0.2, abs=0.1)
        assert res.gamma == pytest.approx(0.1, abs=0.05)
        assert res.df is not None
        assert res.df == pytest.approx(5.0, rel=0.25)
        assert res.n_params == 8

    def test_ar1_nests_the_constant_mean(self) -> None:
        """At phi = 0 the AR(1) likelihood is the constant-mean one, so the
        AR(1) fit can only do better -- and on white-noise means barely does."""
        x = simulate_garch(4000, mu=0.1, seed=30)
        const, ar1 = fit_garch(x), fit_garch(x, mean="ar1")
        assert ar1.loglik >= const.loglik - 1e-6
        assert abs(ar1.phi) < 0.05
        lay = _Layout("garch", "ar1", "normal")
        y = x / x.std()
        c = const
        params = np.array([c.mu / x.std(), 0.0, c.omega / x.var(), c.alpha, c.beta])
        nll = _neg_loglik(params, y, "normal", 1.0, lay)
        assert -nll - x.size * np.log(x.std()) == pytest.approx(const.loglik, rel=1e-12)

    def test_zero_mean_estimates_no_mean(self) -> None:
        x = simulate_garch(3000, mu=0.0, seed=31)
        res = fit_garch(x, mean="zero")
        assert res.mu == 0.0
        assert res.n_params == 3
        assert np.allclose(res.sigma * res.resid, x, rtol=1e-10)
        assert np.all(res.forecast_mean(5) == 0.0)

    def test_residuals_reproduce_the_series(self) -> None:
        x = simulate_model(2000, mu=0.1, phi=0.5, theta=-0.2, seed=32)
        res = fit_garch(x, mean="arma11", vol="gjr")
        assert res.cond_mean is not None
        assert np.allclose(res.cond_mean + res.sigma * res.resid, x, rtol=1e-10)
        # And the conditional mean is the ARMA recursion on the data.
        eps = x - res.cond_mean
        manual = res.mu + res.phi * x[:-1] + res.theta * eps[:-1]
        assert np.allclose(res.cond_mean[1:], manual, rtol=1e-10)

    def test_is_exactly_scale_equivariant(self) -> None:
        x = simulate_model(3000, mu=0.1, phi=0.4, theta=0.1, seed=33)
        a, b = fit_garch(x, mean="arma11"), fit_garch(2.0**5 * x, mean="arma11")
        assert b.phi == pytest.approx(a.phi, rel=1e-10)
        assert b.theta == pytest.approx(a.theta, rel=1e-10)
        assert b.mu == pytest.approx(2.0**5 * a.mu, rel=1e-10)

    def test_forecast_mean(self) -> None:
        x = simulate_model(3000, mu=0.1, phi=0.5, theta=0.2, seed=34)
        res = fit_garch(x, mean="arma11")
        eps_n = res.resid[-1] * res.sigma[-1]
        m = res.forecast_mean(100)
        assert m[0] == pytest.approx(res.mu + res.phi * x[-1] + res.theta * eps_n, rel=1e-10)
        assert m[1] == pytest.approx(res.mu + res.phi * m[0], rel=1e-12)
        assert m[-1] == pytest.approx(res.unconditional_mean, rel=1e-10)
        const = fit_garch(x)
        assert np.all(const.forecast_mean(4) == const.mu)
        with pytest.raises(ValueError, match="horizon must be"):
            res.forecast_mean(0)

    def test_simulation_follows_the_mean_forecast(self) -> None:
        """The simulated paths' average is the ARMA mean forecast."""
        x = simulate_model(3000, mu=0.1, phi=0.6, theta=0.2, seed=35)
        res = fit_garch(x, mean="arma11")
        model = CopulaGarch([res, res], rc.IndependenceCopula(2), innovations="parametric")
        paths = model.simulate(horizon=8, n=100_000, random_state=0)[:, :, 0]
        se = paths.std(axis=0) / np.sqrt(paths.shape[0])
        assert np.all(np.abs(paths.mean(axis=0) - res.forecast_mean(8)) < 5 * se)
        # The AR term makes the variance of the *return* exceed that of the shock.
        assert paths[:, -1].var() > res.forecast_variance(8)[-1]

    def test_rejects_unknown_options(self) -> None:
        x = simulate_garch(200, seed=36)
        with pytest.raises(ValueError, match="vol must be"):
            fit_garch(x, vol="egarch")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="mean must be"):
            fit_garch(x, mean="ar2")  # type: ignore[arg-type]


class TestCopulaGarchOptions:
    def test_fit_passes_vol_and_mean_through(self) -> None:
        cols = [
            simulate_model(2500, mu=0.05, phi=0.3, alpha=0.03, gamma=0.12, beta=0.85, seed=s)
            for s in (40, 41)
        ]
        model = CopulaGarch.fit(
            np.column_stack(cols), rc.GaussianCopula(0.0, dim=2), vol="gjr", mean="ar1"
        )
        for m in model.margins:
            assert m.vol == "gjr"
            assert m.mean == "ar1"
            assert m.gamma > 0.03
            assert m.phi == pytest.approx(0.3, abs=0.08)
        summary = model.summary()
        assert list(summary.columns) == [
            "mu",
            "phi",
            "theta",
            "omega",
            "alpha",
            "beta",
            "gamma",
            "df",
            "persistence",
            "half_life",
            "loglik",
        ]
        paths = model.simulate(horizon=3, n=200, random_state=0)
        assert paths.shape == (200, 3, 2)
        assert np.all(np.isfinite(paths))
        risk = model.forecast_risk(horizon=2, n=2000, random_state=0)
        assert risk["expected_shortfall"] > risk["var"]

    def test_gjr_simulation_reacts_to_the_last_shock(self) -> None:
        """After a fall the simulated next-day volatility is higher than after
        a rally of the same size -- only under GJR."""
        import dataclasses

        res = fit_garch(simulate_model(3000, alpha=0.03, gamma=0.15, beta=0.85, seed=42), vol="gjr")
        down = dataclasses.replace(res, resid=np.append(res.resid[:-1], -3.0))
        up = dataclasses.replace(res, resid=np.append(res.resid[:-1], 3.0))
        cop = rc.IndependenceCopula(2)
        sd_down = CopulaGarch([down, down], cop).simulate(1, 40_000, random_state=0)[:, 0, 0].std()
        sd_up = CopulaGarch([up, up], cop).simulate(1, 40_000, random_state=0)[:, 0, 0].std()
        assert sd_down == pytest.approx(down.forecast_vol(1)[0], rel=0.03)
        assert sd_up == pytest.approx(up.forecast_vol(1)[0], rel=0.03)
        assert sd_down > 1.2 * sd_up


# ---------------------------------------------------------------------------
# Parity with R's rugarch
# ---------------------------------------------------------------------------

GOLDEN = Path(__file__).parent / "golden" / "garch.json"
_GOLDEN = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
_CASES = sorted(k for k in _GOLDEN if not k.startswith("_"))


@pytest.mark.golden
@pytest.mark.skipif(not _CASES, reason="tests/golden/garch.json not generated")
@pytest.mark.parametrize("case", _CASES)
def test_matches_rugarch(case: str) -> None:
    """Same series, same model: parameters to 1e-3, log-likelihood to 0.1.

    rugarch states the ARMA mean around its long-run level, so its ``mu`` maps
    to ``mu * (1 - ar1)`` here. It starts the variance recursion at the mean
    squared residual rather than the sample variance, which moves the
    log-likelihood by a few hundredths at most (observed: 1e-4 to 0.05) and the
    parameters by about 1e-4.
    """
    c = _GOLDEN[case]
    x = np.asarray(c["x"], dtype=float)
    coef = c["coef"]
    vol = "gjr" if c["model"] == "gjrGARCH" else "garch"
    mean = {(0, 0): "constant", (1, 0): "ar1", (1, 1): "arma11"}[tuple(c["arma"])]
    res = fit_garch(x, dist="t" if c["dist"] == "std" else "normal", vol=vol, mean=mean)

    ar = coef.get("ar1", 0.0)
    assert res.mu == pytest.approx(coef["mu"] * (1.0 - ar), abs=1e-3)
    assert res.unconditional_mean == pytest.approx(coef["mu"], abs=1e-3)
    assert res.phi == pytest.approx(ar, abs=1e-3)
    assert res.theta == pytest.approx(coef.get("ma1", 0.0), abs=1e-3)
    assert res.omega == pytest.approx(coef["omega"], abs=1e-3)
    assert res.alpha == pytest.approx(coef["alpha1"], abs=1e-3)
    assert res.beta == pytest.approx(coef["beta1"], abs=1e-3)
    assert res.gamma == pytest.approx(coef.get("gamma1", 0.0), abs=1e-3)
    if "shape" in coef:
        assert res.df == pytest.approx(coef["shape"], rel=1e-3)
    assert res.loglik == pytest.approx(c["loglik"], abs=0.1)
    # Past the start-up the filtered volatilities coincide.
    assert np.allclose(res.sigma[-100:], c["sigma_tail"], rtol=2e-3)
