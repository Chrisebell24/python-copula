"""Tests for the copula-GARCH model.

There is no R ``copula`` oracle for this -- R's copula-GARCH vignette delegates
the marginal models to ``rugarch``. Validation is therefore against properties
that hold exactly:

* **Parameter recovery** from series simulated with known GARCH parameters.
* **Scale equivariance**: GARCH is exactly equivariant under ``x -> c*x``, which
  pins the internal rescaling used to keep the optimiser well conditioned.
* **The recursion itself**, against a literal Python loop.
* **Filtering removes volatility clustering** -- the reason for the first step.
* **The copula is recovered in the innovations** even when the raw returns share
  a volatility regime that makes them look dependent when they are not.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from scipy import signal, stats

import rcopula as rc
from rcopula.garch import CopulaGarch, GarchResult, _filter_variance, fit_garch


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
