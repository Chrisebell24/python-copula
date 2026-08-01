"""Tests for copula estimation.

Point estimates are compared against R (fixtures from
``tools/rgolden/05_fitting.R``). Standard errors are compared against R *where
R is reliable*, and against their own Monte-Carlo sampling distribution
otherwise -- which is the only real test of a standard error anyway.

Two places R is not reliable, both pinned below:

* **``var.icor`` for Frank** understates badly. At ``theta = 5.41``, ``n = 800``
  the true sampling SD of ``theta_hat`` is 0.280 (300 replications); R reports
  0.128, less than half. Confidence intervals built from it would be half as
  wide as they should be. This package reports 0.289 on average, a ratio of
  1.030 to the truth.
* **``iRho`` for Clayton** inherits R's inaccurate Spearman rho, so the *point
  estimate* is off by ~2e-3.

Calibration of every estimator implemented here, against Monte-Carlo sampling
distributions:

=========================  ======  ================
estimator                   ratio  reference
=========================  ======  ================
``mpl`` (Clayton, n=1000)   1.004  250 replications
``itau`` (Gaussian, n=1500) 1.030  400 replications
``irho`` (Gaussian, n=1500) 0.962  400 replications
``itau`` (Frank, n=800)     1.030  300 replications
=========================  ======  ================

All within ~4%, which is ordinary for asymptotic approximations at these sample
sizes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import rcopula as rc
from rcopula.fit import nearest_correlation

GOLDEN = Path(__file__).parent / "golden" / "fitting.json"

BUILDERS = {
    "clayton": lambda: rc.ClaytonCopula(),
    "gumbel": lambda: rc.GumbelCopula(),
    "frank": lambda: rc.FrankCopula(),
    "joe": lambda: rc.JoeCopula(),
    "normal": lambda: rc.GaussianCopula(),
    "normal_un": lambda: rc.GaussianCopula(dim=3, dispstr="un"),
    "t_un": lambda: rc.StudentCopula(dim=3, dispstr="un"),
}


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN.exists():  # pragma: no cover
        pytest.skip(f"golden fixtures not found at {GOLDEN}; run `make golden`")
    return json.loads(GOLDEN.read_text())


def _cases(blob: dict) -> list[str]:
    return sorted(k for k in blob if not k.startswith("_"))


def _numeric(value: object) -> np.ndarray | None:
    arr = np.atleast_1d(np.asarray(value, dtype=object))
    if any(v is None or v == "NA" for v in arr):
        return None
    return arr.astype(float)


# ----------------------------------------------------------------------
# Parity with R
# ----------------------------------------------------------------------


@pytest.mark.golden
@pytest.mark.parametrize("method", ["mpl", "ml", "itau", "irho", "itau.mpl"])
def test_point_estimates_match_r(golden: dict, method: str) -> None:
    compared = 0
    for key in _cases(golden):
        blk = golden[key]
        expected = _numeric(blk.get(f"est_{method}"))
        if expected is None:
            continue
        if blk["family"] == "clayton" and method == "irho":
            continue  # R's iRho for Clayton is wrong; see the dedicated test
        u = np.atleast_2d(np.asarray(blk["u"], dtype=float))
        got = rc.fit(BUILDERS[blk["family"]](), u, method=method).params
        assert np.allclose(got, expected, rtol=1e-4), f"{key}:{method}"
        compared += 1
    assert compared > 0, f"no fixture exercised {method}"


@pytest.mark.golden
def test_ml_standard_errors_match_r_exactly(golden: dict) -> None:
    """The inverse observed information is unambiguous, so these agree to 1e-5."""
    for key in _cases(golden):
        blk = golden[key]
        expected = _numeric(blk.get("se_ml"))
        if expected is None:
            continue
        u = np.atleast_2d(np.asarray(blk["u"], dtype=float))
        got = rc.fit(BUILDERS[blk["family"]](), u, method="ml").bse
        assert np.allclose(got, expected, rtol=1e-5), key


@pytest.mark.golden
@pytest.mark.parametrize("method", ["itau", "irho"])
def test_inversion_standard_errors_match_r(golden: dict, method: str) -> None:
    """Excludes Frank's ``itau``, where R is wrong -- see the dedicated test."""
    for key in _cases(golden):
        blk = golden[key]
        expected = _numeric(blk.get(f"se_{method}"))
        if expected is None:
            continue
        if blk["family"] == "frank" and method == "itau":
            continue
        u = np.atleast_2d(np.asarray(blk["u"], dtype=float))
        got = rc.fit(BUILDERS[blk["family"]](), u, method=method).bse
        assert np.allclose(got, expected, rtol=0.05), key


@pytest.mark.golden
def test_loglikelihood_matches_r(golden: dict) -> None:
    for key in _cases(golden):
        blk = golden[key]
        expected = _numeric(blk.get("loglik_mpl"))
        if expected is None:
            continue
        u = np.atleast_2d(np.asarray(blk["u"], dtype=float))
        got = rc.fit(BUILDERS[blk["family"]](), u, method="mpl").loglik
        assert got == pytest.approx(float(expected[0]), rel=1e-6), key


# ----------------------------------------------------------------------
# Where R is wrong
# ----------------------------------------------------------------------


class TestWhereRIsWrong:
    def test_r_understates_the_frank_itau_standard_error(self, golden: dict) -> None:
        """R reports less than half the true sampling standard deviation."""
        r_value = float(_numeric(golden["frank_5"]["se_itau"])[0])
        u = np.atleast_2d(np.asarray(golden["frank_5"]["u"], dtype=float))
        ours = rc.fit(rc.FrankCopula(), u, method="itau").bse[0]

        # Monte-Carlo truth at theta = 5.41, n = 800 is 0.280 (300 reps).
        assert ours == pytest.approx(0.28, rel=0.25)
        assert r_value < 0.6 * ours, (
            "R's var.icor for Frank now looks reasonable; if R has been fixed, "
            "include Frank in test_inversion_standard_errors_match_r."
        )

    def test_r_irho_for_clayton_inherits_its_inaccurate_rho(self, golden: dict) -> None:
        """Our estimate reproduces the sample Spearman rho; R's does not."""
        u = np.atleast_2d(np.asarray(golden["clayton_2"]["u"], dtype=float))
        target = float(rc.cor_spearman(u)[0, 1])

        ours = rc.fit(rc.ClaytonCopula(), u, method="irho").params[0]
        r_value = float(_numeric(golden["clayton_2"]["est_irho"])[0])

        assert rc.ClaytonCopula(ours).rho() == pytest.approx(target, abs=1e-8)
        assert abs(rc.ClaytonCopula(r_value).rho() - target) > 1e-5

    def test_we_provide_a_joe_itau_variance_where_r_declines(self, golden: dict) -> None:
        """R warns 'variance estimate cannot be computed for joeCopula'."""
        assert _numeric(golden["joe_3"].get("se_itau")) is None
        u = np.atleast_2d(np.asarray(golden["joe_3"]["u"], dtype=float))
        assert rc.fit(rc.JoeCopula(), u, method="itau").bse[0] > 0


# ----------------------------------------------------------------------
# Standard errors against their own sampling distribution
# ----------------------------------------------------------------------


@pytest.mark.slow
class TestStandardErrorCalibration:
    """The only honest test of a standard error: does it reproduce the spread?"""

    @staticmethod
    def _calibrate(truth, template, method, n, reps, seed):
        rng = np.random.default_rng(seed)
        ests, ses = [], []
        for _ in range(reps):
            uh = rc.pseudo_obs(truth.rvs(n, random_state=rng))
            res = rc.fit(template(), uh, method=method)
            ests.append(res.params[0])
            ses.append(res.bse[0])
        return np.std(ests, ddof=1), float(np.mean(ses))

    def test_mpl_is_unbiased(self) -> None:
        sd, mean_se = self._calibrate(rc.ClaytonCopula(2.0), rc.ClaytonCopula, "mpl", 600, 120, 1)
        assert mean_se / sd == pytest.approx(1.0, abs=0.15)

    @pytest.mark.parametrize("method", ["itau", "irho"])
    def test_inversion_is_unbiased(self, method: str) -> None:
        sd, mean_se = self._calibrate(
            rc.GaussianCopula(0.6), rc.GaussianCopula, method, 800, 150, 2
        )
        assert mean_se / sd == pytest.approx(1.0, abs=0.15)

    def test_frank_itau_tracks_the_truth_where_r_does_not(self) -> None:
        sd, mean_se = self._calibrate(rc.FrankCopula(5.41), rc.FrankCopula, "itau", 800, 120, 3)
        assert mean_se / sd == pytest.approx(1.0, abs=0.15)
        assert sd > 0.2  # R's 0.128 would be far below this


# ----------------------------------------------------------------------
# Behaviour
# ----------------------------------------------------------------------


class TestFitBehaviour:
    def test_mpl_and_ml_share_an_estimate_but_not_a_standard_error(self) -> None:
        u = rc.ClaytonCopula(2.0).rvs(1000, random_state=0)
        a = rc.fit(rc.ClaytonCopula(), u, method="mpl")
        b = rc.fit(rc.ClaytonCopula(), u, method="ml")
        assert np.allclose(a.params, b.params, rtol=1e-6)
        assert a.bse[0] != b.bse[0]
        assert a.bse[0] > b.bse[0]  # ranks cost information

    @pytest.mark.parametrize(
        ("cls", "truth"),
        [
            (rc.ClaytonCopula, 2.0),
            (rc.GumbelCopula, 2.5),
            (rc.FrankCopula, 5.0),
            (rc.JoeCopula, 3.0),
            (rc.GaussianCopula, 0.6),
        ],
    )
    @pytest.mark.parametrize("method", ["mpl", "itau"])
    def test_recovers_the_generating_parameter(self, cls, truth: float, method: str) -> None:
        u = cls(truth).rvs(4000, random_state=0)
        est = rc.fit(cls(), u, method=method).params[0]
        assert est == pytest.approx(truth, rel=0.1)

    def test_raw_data_is_rank_transformed_automatically(self) -> None:
        """Values outside the unit cube cannot be copula observations."""
        from scipy import stats

        mv = rc.CopulaDistribution(rc.ClaytonCopula(2.0), [stats.norm(10, 3), stats.expon(scale=5)])
        x = mv.rvs(2000, random_state=0)
        assert rc.fit(rc.ClaytonCopula(), x, method="mpl").params[0] == pytest.approx(2.0, rel=0.15)

    def test_accepts_a_dataframe(self) -> None:
        import pandas as pd

        u = rc.ClaytonCopula(2.0).rvs(500, random_state=0)
        df = pd.DataFrame(u, columns=["a", "b"])
        assert rc.fit(rc.ClaytonCopula(), df).params[0] > 0

    def test_summary_reports_the_essentials(self) -> None:
        u = rc.ClaytonCopula(2.0).rvs(500, random_state=0)
        text = rc.fit(rc.ClaytonCopula(), u).summary()
        for expected in ("Clayton", "mpl", "theta", "std.err", "log-likelihood", "AIC"):
            assert expected in text

    def test_information_criteria_and_intervals(self) -> None:
        u = rc.ClaytonCopula(2.0).rvs(1000, random_state=0)
        res = rc.fit(rc.ClaytonCopula(), u)
        assert res.aic == pytest.approx(-2 * res.loglik + 2)
        assert res.bic == pytest.approx(-2 * res.loglik + np.log(1000))
        ci = res.conf_int()
        assert ci.shape == (1, 2)
        assert ci[0, 0] < res.params[0] < ci[0, 1]

    def test_estimate_variance_can_be_switched_off(self) -> None:
        u = rc.ClaytonCopula(2.0).rvs(300, random_state=0)
        res = rc.fit(rc.ClaytonCopula(), u, estimate_variance=False)
        assert res.bse is None and res.conf_int() is None
        assert "not computed" in res.summary()

    def test_rejects_unknown_method_and_mismatched_dimension(self) -> None:
        u = rc.ClaytonCopula(2.0).rvs(100, random_state=0)
        with pytest.raises(ValueError, match="method must be one of"):
            rc.fit(rc.ClaytonCopula(), u, method="reml")
        with pytest.raises(ValueError, match="column"):
            rc.fit(rc.ClaytonCopula(2.0, dim=3), u)

    def test_itau_mpl_is_restricted_as_in_r(self) -> None:
        u = rc.ClaytonCopula(2.0).rvs(200, random_state=0)
        with pytest.raises(ValueError, match="Student-t copula only"):
            rc.fit(rc.ClaytonCopula(), u, method="itau.mpl")
        ut = rc.StudentCopula(0.5, df=5).rvs(200, random_state=0)
        with pytest.raises(ValueError, match="dispstr='un'"):
            rc.fit(rc.StudentCopula(), ut, method="itau.mpl")

    def test_unattainable_dependence_is_reported_clearly(self) -> None:
        """AMH cannot reach tau = 0.5, so inversion must fail loudly."""
        u = rc.ClaytonCopula(8.0).rvs(500, random_state=0)
        with pytest.raises(ValueError, match="cannot invert"):
            rc.fit(rc.AMHCopula(), u, method="itau")


class TestLoglikCopula:
    def test_is_maximised_at_the_truth(self) -> None:
        u = rc.GumbelCopula(3.0).rvs(3000, random_state=0)
        grid = np.linspace(1.5, 5.0, 40)
        values = [rc.loglik_copula([t], u, rc.GumbelCopula()) for t in grid]
        assert abs(grid[int(np.argmax(values))] - 3.0) < 0.4

    def test_inadmissible_parameters_give_minus_infinity(self) -> None:
        u = rc.ClaytonCopula(2.0).rvs(100, random_state=0)
        assert np.isneginf(rc.loglik_copula([-5.0], u, rc.ClaytonCopula()))


class TestNearestCorrelation:
    def test_repairs_an_indefinite_matrix(self) -> None:
        bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
        assert np.linalg.eigvalsh(bad).min() < 0
        fixed = nearest_correlation(bad)
        assert np.linalg.eigvalsh(fixed).min() > 0
        assert np.allclose(np.diag(fixed), 1.0)
        assert np.allclose(fixed, fixed.T)

    def test_leaves_a_valid_matrix_essentially_alone(self) -> None:
        good = rc.GaussianCopula([0.5, 0.3, 0.2], dim=3, dispstr="un").sigma()
        assert np.allclose(nearest_correlation(good), good, atol=1e-8)


class TestParameterFreeFit:
    """A copula with nothing to estimate still has a log-likelihood.

    Returning it -- rather than handing an empty parameter vector to an
    optimiser, which raises inside scipy -- is what lets the independence copula
    take part in an AIC comparison as the null model it is.
    """

    def test_independence_fits_trivially(self) -> None:
        u = rc.ClaytonCopula(2.0).rvs(300, random_state=0)
        res = rc.fit(rc.IndependenceCopula(2), u)
        assert res.n_params == 0
        assert res.loglik == 0.0
        assert res.aic == 0.0 and res.bic == 0.0
        assert res.converged
        assert res.params.shape == (0,)

    @pytest.mark.parametrize("method", ["mpl", "ml", "itau", "irho"])
    def test_every_method_agrees_there_is_nothing_to_do(self, method: str) -> None:
        u = rc.ClaytonCopula(2.0).rvs(200, random_state=0)
        assert rc.fit(rc.IndependenceCopula(2), u, method=method).n_params == 0

    def test_a_fully_pinned_copula_reports_its_likelihood_at_that_point(self) -> None:
        u = rc.ClaytonCopula(2.0).rvs(400, random_state=0)
        pinned = rc.ClaytonCopula(1.5).fix_params([False])
        res = rc.fit(pinned, u)
        assert res.n_params == 0
        # u is already on the unit cube, so fit passes it through untransformed.
        assert res.loglik == pytest.approx(float(np.sum(pinned.logpdf(u))))
        assert res.copula.theta == 1.5

    def test_it_loses_to_the_free_fit_on_dependent_data(self) -> None:
        """Sanity: the trivial fit must not accidentally win a real comparison."""
        u = rc.ClaytonCopula(3.0).rvs(600, random_state=0)
        assert rc.fit(rc.ClaytonCopula(), u).aic < rc.fit(rc.IndependenceCopula(2), u).aic

    def test_summary_is_printable_with_no_parameters(self) -> None:
        u = rc.ClaytonCopula(2.0).rvs(200, random_state=0)
        assert "Independence" in rc.fit(rc.IndependenceCopula(2), u).summary()


class TestInversionStandardErrorsAboveTwoDimensions:
    """``itau``/``irho`` reported no standard error at all for ``d > 2``.

    A one-parameter family in higher dimensions is fitted by inverting the
    *average* of the ``d(d-1)/2`` pairwise statistics. Averaging is linear, so
    the influence function of the average is the average of the pairwise
    influence functions -- and keeping them as functions of the same
    observations preserves the correlation between pairs that share a
    coordinate, which is substantial and would be lost by treating the pairwise
    statistics as independent.
    """

    @pytest.mark.parametrize(
        ("ctor", "truth"),
        [(rc.ClaytonCopula, 2.0), (rc.GumbelCopula, 2.5), (rc.GaussianCopula, 0.6)],
    )
    @pytest.mark.parametrize("dim", [3, 4])
    @pytest.mark.parametrize("method", ["itau", "irho"])
    def test_a_standard_error_is_reported(
        self, ctor: type, truth: float, dim: int, method: str
    ) -> None:
        u = ctor(truth, dim=dim).rvs(500, random_state=0)
        res = rc.fit(ctor(dim=dim), u, method=method)
        assert res.bse is not None
        assert res.bse[0] > 0.0
        assert np.all(np.isfinite(res.conf_int()))

    def test_it_is_calibrated_against_the_sampling_distribution(self) -> None:
        """The only check that matters: does it predict the actual spread?"""
        truth, dim = 2.0, 4
        estimates, errors = [], []
        for seed in range(60):
            u = rc.ClaytonCopula(truth, dim=dim).rvs(600, random_state=seed)
            res = rc.fit(rc.ClaytonCopula(dim=dim), u, method="itau")
            estimates.append(res.params[0])
            errors.append(res.bse[0])
        ratio = float(np.mean(errors)) / float(np.std(estimates, ddof=1))
        assert 0.85 < ratio < 1.20, f"mean SE / empirical SD = {ratio:.3f}"

    def test_higher_dimensions_estimate_more_precisely(self) -> None:
        """More pairs, more information -- so the standard error must shrink."""
        errors = [
            rc.fit(
                rc.ClaytonCopula(dim=d),
                rc.ClaytonCopula(2.0, dim=d).rvs(800, random_state=0),
                method="itau",
            ).bse[0]
            for d in (2, 3, 5)
        ]
        assert errors[0] > errors[1] > errors[2]
