"""Tests for :mod:`rcopula.fit.mvdc`.

Two claims in the module docstring are worth checking rather than trusting.

The first is that **full maximum likelihood buys little over IFM**. That is
asserted in the literature and repeated everywhere; here it is measured, on data
where the truth is known.

The second is about ``margin_kwargs``: leaving a location parameter free when
the family requires it to be zero is called the commonest cause of an
implausible fit. Pinning it turns out to recover nearly all of the gap between
IFM and full ML, which makes the advice concrete rather than folklore.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import rcopula as rc
from rcopula.fit.mvdc import fit_joint


@pytest.fixture(scope="module")
def sample() -> tuple[np.ndarray, rc.CopulaDistribution]:
    truth = rc.CopulaDistribution(
        rc.ClaytonCopula(2.0), [stats.norm(1.0, 2.0), stats.expon(scale=3.0)]
    )
    return truth.rvs(3000, random_state=0), truth


def _template() -> rc.CopulaDistribution:
    return rc.CopulaDistribution(rc.ClaytonCopula(1.0), [stats.norm(), stats.expon()])


class TestRecovery:
    @pytest.mark.parametrize("method", ["ifm", "ml"])
    def test_it_recovers_the_margins(self, method: str, sample) -> None:
        x, _ = sample
        result = fit_joint(_template(), x, method=method)
        (mean, sd), (_, scale) = result.margin_params
        assert mean == pytest.approx(1.0, abs=0.1)
        assert sd == pytest.approx(2.0, abs=0.1)
        assert scale == pytest.approx(3.0, abs=0.15)

    @pytest.mark.parametrize("method", ["ifm", "ml"])
    def test_it_recovers_the_copula(self, method: str, sample) -> None:
        x, _ = sample
        result = fit_joint(_template(), x, method=method)
        assert float(result.copula.params[0]) == pytest.approx(2.0, abs=0.25)

    def test_the_result_is_a_usable_distribution(self, sample) -> None:
        x, _ = sample
        result = fit_joint(_template(), x)
        draws = result.distribution.rvs(500, random_state=0)
        assert draws.shape == (500, 2)
        # The refitted model reproduces the dependence it was fitted to.
        assert abs(float(rc.cor_kendall(draws)[0, 1]) - float(rc.cor_kendall(x)[0, 1])) < 0.05

    def test_a_gaussian_copula_with_normal_margins(self, sample) -> None:
        del sample
        truth = rc.CopulaDistribution(rc.GaussianCopula(0.6), [stats.norm(0, 1), stats.norm(2, 3)])
        x = truth.rvs(3000, random_state=1)
        template = rc.CopulaDistribution(rc.GaussianCopula(0.0), [stats.norm()] * 2)
        result = fit_joint(template, x)
        assert float(result.copula.params[0]) == pytest.approx(0.6, abs=0.04)
        assert result.margin_params[1][1] == pytest.approx(3.0, abs=0.15)

    def test_three_dimensions(self) -> None:
        truth = rc.CopulaDistribution(
            rc.GumbelCopula(2.0, dim=3), [stats.norm(), stats.norm(1.0), stats.norm(-1.0, 2.0)]
        )
        x = truth.rvs(2000, random_state=0)
        template = rc.CopulaDistribution(rc.GumbelCopula(1.5, dim=3), [stats.norm()] * 3)
        result = fit_joint(template, x)
        assert float(result.copula.params[0]) == pytest.approx(2.0, abs=0.15)
        assert len(result.margin_params) == 3


class TestTheClaimsInTheDocstring:
    def test_full_ml_buys_little_over_ifm(self, sample) -> None:
        x, _ = sample
        ifm = fit_joint(_template(), x, method="ifm")
        ml = fit_joint(_template(), x, method="ml")
        # ML maximises the same objective over a superset, so it cannot be worse.
        assert ml.loglik >= ifm.loglik - 1e-6
        # And the gain is small relative to the likelihood itself.
        assert (ml.loglik - ifm.loglik) < 0.01 * abs(ifm.loglik)

    def test_pinning_a_location_recovers_most_of_the_gap(self, sample) -> None:
        # scipy's expon.fit slides `loc` to the sample minimum when it is free.
        # That is the right MLE for the margin alone and the wrong model here,
        # and it is what margin_kwargs is for.
        x, _ = sample
        free = fit_joint(_template(), x, method="ifm")
        pinned = fit_joint(_template(), x, method="ifm", margin_kwargs=[{}, {"floc": 0}])
        full = fit_joint(_template(), x, method="ml")
        assert pinned.margin_params[1][0] == 0.0
        assert pinned.loglik > free.loglik
        # Within a hair of what joint optimisation achieves, for a fraction of
        # the work.
        assert (full.loglik - pinned.loglik) < 0.1 * (full.loglik - free.loglik)

    def test_a_free_boundary_shows_up_in_the_count(self, sample) -> None:
        x, _ = sample
        free = fit_joint(_template(), x, method="ifm")
        pinned = fit_joint(_template(), x, method="ifm", margin_kwargs=[{}, {"floc": 0}])
        # The sample minimum maps to F(x) = 0 exactly when loc is free.
        assert free.n_at_boundary >= 1
        assert pinned.n_at_boundary == 0

    def test_the_log_likelihood_stays_finite_at_the_boundary(self, sample) -> None:
        # Without the nudge this is -inf, and a perfectly good fit reports an
        # infinite likelihood.
        x, _ = sample
        assert np.isfinite(fit_joint(_template(), x, method="ifm").loglik)


class TestResultObject:
    def test_information_criteria_count_every_parameter(self, sample) -> None:
        x, _ = sample
        result = fit_joint(_template(), x)
        assert result.n_params == 5  # two per margin, one copula
        assert result.aic == pytest.approx(2 * 5 - 2 * result.loglik)
        assert result.bic == pytest.approx(5 * np.log(3000) - 2 * result.loglik)

    def test_the_likelihood_splits_into_margins_and_dependence(self, sample) -> None:
        x, _ = sample
        result = fit_joint(_template(), x)
        assert result.loglik == pytest.approx(result.marginal_loglik + result.dependence_loglik)
        # Dependence is a gain over independence, so it is positive here.
        assert result.dependence_loglik > 0

    def test_dependence_loglik_ranks_families_sensibly(self, sample) -> None:
        # The joint figure is dominated by the margins and cannot rank copulas;
        # the dependence part can.
        x, _ = sample
        scores = {}
        for family in (rc.ClaytonCopula(1.0), rc.GumbelCopula(1.5), rc.FrankCopula(1.0)):
            template = rc.CopulaDistribution(family, [stats.norm(), stats.expon()])
            scores[type(family).__name__] = fit_joint(template, x).dependence_loglik
        assert max(scores, key=lambda k: scores[k]) == "ClaytonCopula"

    def test_summary_reports_the_pieces(self, sample) -> None:
        x, _ = sample
        text = fit_joint(_template(), x).summary()
        for expected in ("Joint fit by IFM", "margin 0", "copula", "of which dependence"):
            assert expected in text


class TestValidation:
    def test_rejects_a_dimension_mismatch(self, sample) -> None:
        x, _ = sample
        with pytest.raises(ValueError, match="columns"):
            fit_joint(rc.CopulaDistribution(rc.ClaytonCopula(1.0, dim=3), [stats.norm()] * 3), x)

    def test_rejects_an_unknown_method(self, sample) -> None:
        x, _ = sample
        with pytest.raises(ValueError, match="'ifm' or 'ml'"):
            fit_joint(_template(), x, method="mpl")

    def test_rejects_the_wrong_number_of_margin_kwargs(self, sample) -> None:
        x, _ = sample
        with pytest.raises(ValueError, match="margin_kwargs must have 2"):
            fit_joint(_template(), x, margin_kwargs=[{}])

    def test_a_margin_that_cannot_be_refitted_says_so(self, sample) -> None:
        x, _ = sample

        class Custom:
            def cdf(self, v):
                return stats.norm.cdf(v)

            def pdf(self, v):
                return stats.norm.pdf(v)

            def ppf(self, q):
                return stats.norm.ppf(q)

        template = rc.CopulaDistribution(rc.ClaytonCopula(1.0), [stats.norm(), Custom()])
        with pytest.raises(TypeError, match="cannot be refitted"):
            fit_joint(template, x)
