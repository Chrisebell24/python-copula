"""Tests for :mod:`rcopula.dynamic`.

Two things here are load-bearing and everything else is scaffolding around them.

The first is the **fast-path cross-check**: the module carries a hand-written
vectorised log-density for five families, purely for speed. Every one is
compared against the family it is supposed to mirror, over the full range of the
link, at machine precision. If a formula ever drifts from the family, this
fails, and no result silently changes.

The second is the **DCC likelihood cross-check**: ``_dcc_loglik`` is written
directly rather than through a copula object, again for speed. It is compared
against ``GaussianCopula.logpdf`` and ``StudentCopula.logpdf`` at a fixed
correlation, which is the only place those two implementations can disagree.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import linalg, stats

import rcopula as rc
from rcopula.dynamic import (
    DccResult,
    DynamicCopula,
    DynamicFitResult,
    _dcc_filter,
    _dcc_loglik,
    _default_link,
    _forcing_values,
    _Link,
    _moving_average,
    fit_dcc,
    fit_dynamic,
)


@pytest.fixture(scope="module")
def sample() -> np.ndarray:
    return np.clip(rc.GaussianCopula(0.5).rvs(600, random_state=0), 1e-9, 1 - 1e-9)


class TestLink:
    def test_maps_onto_the_open_interval(self) -> None:
        link = _Link(-1.0, 1.0)
        x = np.array([-1e6, -20.0, -1.0, 0.0, 1.0, 20.0, 1e6])
        values = link(x)
        assert np.all(values > -1.0)
        assert np.all(values < 1.0)

    def test_never_returns_the_endpoint(self) -> None:
        # tanh saturates to exactly 1.0 past |x| ~ 19 in double precision, and a
        # correlation of exactly 1 fails the Gaussian copula's Cholesky.
        link = _Link(-1.0, 1.0)
        assert float(link(50.0)) < 1.0
        assert float(link(-50.0)) > -1.0

    def test_round_trip(self) -> None:
        link = _Link(1.0, 26.0)
        for theta in (1.5, 3.0, 12.0, 25.0):
            assert float(link(link.inverse(theta))) == pytest.approx(theta, rel=1e-9)

    def test_standardise_is_the_identity_on_minus_one_to_one(self) -> None:
        # Patton's own case: beta multiplies the lagged correlation itself.
        link = _Link(-1.0, 1.0)
        for theta in (-0.9, -0.1, 0.0, 0.4, 0.95):
            assert link.standardise(theta) == pytest.approx(theta)

    def test_standardise_rescales_a_wide_range(self) -> None:
        link = _Link(1.0, 26.0)
        assert link.standardise(1.0) == pytest.approx(-1.0)
        assert link.standardise(26.0) == pytest.approx(1.0)
        assert link.standardise(13.5) == pytest.approx(0.0)

    def test_monotone(self) -> None:
        link = _Link(0.0, 5.0)
        values = link(np.linspace(-10, 10, 200))
        assert np.all(np.diff(values) >= 0)

    @pytest.mark.parametrize(
        ("copula", "expected"),
        [
            (rc.GaussianCopula(0.0), (-1.0, 1.0)),
            (rc.ClaytonCopula(1.0), (-1.0, 24.0)),
            (rc.GumbelCopula(2.0), (1.0, 26.0)),
        ],
    )
    def test_default_range_from_the_family(
        self, copula: rc.Copula, expected: tuple[float, float]
    ) -> None:
        link = _default_link(copula, 0, None)
        assert (link.lower, link.upper) == pytest.approx(expected)

    def test_frank_gets_a_two_sided_range(self) -> None:
        link = _default_link(rc.FrankCopula(2.0), 0, None)
        assert link.lower < -30 and link.upper > 30

    def test_explicit_bounds_win(self) -> None:
        link = _default_link(rc.ClaytonCopula(1.0), 0, (0.5, 4.0))
        assert (link.lower, link.upper) == (0.5, 4.0)

    def test_rejects_a_reversed_interval(self) -> None:
        with pytest.raises(ValueError, match="increasing"):
            _default_link(rc.ClaytonCopula(1.0), 0, (4.0, 0.5))


#: Every family with a hand-written fast density, over the full link range.
_FAST_CASES = [
    (rc.GaussianCopula(0.3), np.linspace(-0.995, 0.995, 400)),
    (rc.StudentCopula(0.3, df=4.0), np.linspace(-0.995, 0.995, 400)),
    (rc.StudentCopula(0.3, df=2.5), np.linspace(-0.9, 0.9, 400)),
    (rc.ClaytonCopula(2.0), np.linspace(1e-3, 24.0, 400)),
    (rc.GumbelCopula(2.0), np.linspace(1.0, 26.0, 400)),
    (rc.FrankCopula(2.0), np.linspace(-40.0, 40.0, 400)),
]


class TestFastDensities:
    """The reason the module is fast, checked against the reason it is right."""

    @pytest.mark.parametrize(("family", "path"), _FAST_CASES, ids=lambda v: getattr(v, "shape", ""))
    def test_matches_the_family_it_mirrors(
        self, family: rc.Copula, path: np.ndarray, sample: np.ndarray
    ) -> None:
        u = sample[: path.size]
        model = DynamicCopula(family, coefficients=(0.0, 0.0, 0.0))
        fast = model._logpdf_rows(u, path)
        generic = np.array(
            [family._logpdf(u[t : t + 1], model._params_at(path[t]))[0] for t in range(path.size)]
        )
        assert np.all(np.isfinite(generic))
        np.testing.assert_allclose(fast, generic, rtol=1e-10, atol=1e-10)

    def test_frank_survives_the_top_of_its_range(self, sample: np.ndarray) -> None:
        # The naive denominator loses eleven digits by theta = 34, which is well
        # inside the link's range, so this is not a hypothetical.
        family = rc.FrankCopula(2.0)
        model = DynamicCopula(family, coefficients=(0.0, 0.0, 0.0))
        path = np.full(50, 38.0)
        fast = model._logpdf_rows(sample[:50], path)
        generic = np.array(
            [family._logpdf(sample[t : t + 1], np.array([38.0]))[0] for t in range(50)]
        )
        np.testing.assert_allclose(fast, generic, rtol=1e-10)

    def test_falls_back_when_the_path_leaves_the_valid_region(self, sample: np.ndarray) -> None:
        # Clayton's fast expression is only written for positive theta. A path
        # that crosses zero must still produce the family's own answer.
        family = rc.ClaytonCopula(1.0)
        model = DynamicCopula(family, coefficients=(0.0, 0.0, 0.0))
        path = np.linspace(-0.5, 2.0, 40)
        values = model._logpdf_rows(sample[:40], path)
        generic = np.array(
            [family._logpdf(sample[t : t + 1], np.array([path[t]]))[0] for t in range(40)]
        )
        np.testing.assert_allclose(values, generic, rtol=1e-10)

    def test_a_family_without_a_fast_path_still_works(self, sample: np.ndarray) -> None:
        family = rc.JoeCopula(2.0)
        model = DynamicCopula(family, coefficients=(0.0, 0.0, 0.0))
        path = np.linspace(1.2, 5.0, 30)
        values = model._logpdf_rows(sample[:30], path)
        assert values.shape == (30,)
        assert np.all(np.isfinite(values))


class TestForcing:
    def test_moving_average_of_the_previous_lags(self) -> None:
        values = np.arange(1.0, 11.0)
        averaged = _moving_average(values, lags=3)
        # Position 5 averages values 3, 4 and 5 -- that is, indices 2, 3, 4.
        assert averaged[5] == pytest.approx(np.mean(values[2:5]))
        assert averaged[9] == pytest.approx(np.mean(values[6:9]))

    def test_presample_is_the_sample_mean(self) -> None:
        values = np.arange(1.0, 11.0)
        averaged = _moving_average(values, lags=3)
        assert np.all(averaged[:3] == pytest.approx(values.mean()))

    def test_lags_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            _moving_average(np.arange(5.0), lags=0)

    def test_normal_product_tracks_dependence(self) -> None:
        strong = rc.GaussianCopula(0.9).rvs(4000, random_state=0)
        weak = rc.GaussianCopula(0.1).rvs(4000, random_state=0)
        assert np.mean(_forcing_values(strong, "normal-product", None)) > np.mean(
            _forcing_values(weak, "normal-product", None)
        )

    def test_absolute_difference_falls_as_dependence_rises(self) -> None:
        # The sign flip that makes a fitted alpha negative for this forcing.
        strong = rc.GaussianCopula(0.9).rvs(4000, random_state=0)
        weak = rc.GaussianCopula(0.1).rvs(4000, random_state=0)
        assert np.mean(_forcing_values(strong, "abs-difference", None)) < np.mean(
            _forcing_values(weak, "abs-difference", None)
        )

    def test_infinite_quantiles_are_clipped(self) -> None:
        u = np.array([[0.0, 0.5], [1.0, 0.5]])
        values = _forcing_values(u, "normal-product", None)
        assert np.all(np.isfinite(values))

    def test_student_forcing_uses_t_quantiles(self) -> None:
        u = np.array([[0.99, 0.99]])
        normal = _forcing_values(u, "normal-product", None)
        heavy = _forcing_values(u, "normal-product", 3.0)
        assert heavy[0] > normal[0]

    def test_unknown_forcing(self) -> None:
        with pytest.raises(ValueError, match="unknown forcing"):
            _forcing_values(np.array([[0.5, 0.5]]), "nonsense", None)


class TestConstruction:
    def test_rejects_a_non_bivariate_family(self) -> None:
        with pytest.raises(ValueError, match="fit_dcc"):
            DynamicCopula(rc.GaussianCopula(0.5, dim=3), coefficients=(0, 0, 0))

    def test_rejects_the_wrong_number_of_coefficients(self) -> None:
        with pytest.raises(ValueError, match="omega, alpha, beta"):
            DynamicCopula(rc.GaussianCopula(0.5), coefficients=(0.1, 0.2))

    def test_rejects_an_unknown_driver(self) -> None:
        with pytest.raises(ValueError, match="patton"):
            DynamicCopula(rc.GaussianCopula(0.5), coefficients=(0, 0, 0), driver="kalman")

    @pytest.mark.parametrize(
        ("family", "expected"),
        [
            (rc.GaussianCopula(0.5), "normal-product"),
            (rc.StudentCopula(0.5, df=5.0), "normal-product"),
            (rc.ClaytonCopula(2.0), "abs-difference"),
            (rc.GumbelCopula(2.0), "abs-difference"),
        ],
    )
    def test_forcing_default_follows_patton(self, family: rc.Copula, expected: str) -> None:
        assert DynamicCopula(family, coefficients=(0, 0, 0)).forcing == expected

    def test_repr_shows_the_coefficients(self) -> None:
        text = repr(DynamicCopula(rc.GaussianCopula(0.5), coefficients=(0.1, 0.2, 0.9)))
        assert "0.1" in text and "0.9" in text and "patton" in text

    def test_with_coefficients_preserves_everything_else(self) -> None:
        model = DynamicCopula(
            rc.ClaytonCopula(2.0),
            coefficients=(0, 0, 0),
            driver="gas",
            lags=4,
            bounds=(0.1, 8.0),
        )
        other = model.with_coefficients((0.5, 0.1, 0.8))
        assert other.driver == "gas"
        assert other.lags == 4
        assert (other.link.lower, other.link.upper) == (0.1, 8.0)


class TestFilter:
    def test_zero_coefficients_pin_the_parameter(self, sample: np.ndarray) -> None:
        model = DynamicCopula(rc.GaussianCopula(0.5), coefficients=(0.0, 0.0, 0.0))
        path = model.filter(sample).path
        # omega = alpha = beta = 0 gives Lambda(0), the centre of the range.
        assert np.allclose(path, 0.0)

    def test_shape_and_finiteness(self, sample: np.ndarray) -> None:
        model = DynamicCopula(rc.GaussianCopula(0.3), coefficients=(0.2, 0.3, 0.7))
        result = model.filter(sample)
        assert result.path.shape == (sample.shape[0],)
        assert result.linked.shape == (sample.shape[0],)
        assert np.all(np.isfinite(result.loglik_contributions))
        assert result.loglik == pytest.approx(result.loglik_contributions.sum())

    def test_path_stays_inside_the_link(self, sample: np.ndarray) -> None:
        model = DynamicCopula(rc.ClaytonCopula(1.0), coefficients=(3.0, 2.0, 0.99))
        path = model.filter(sample).path
        assert np.all(path > model.link.lower)
        assert np.all(path < model.link.upper)

    def test_patton_reproduces_its_own_recursion_by_hand(self, sample: np.ndarray) -> None:
        omega, alpha, beta = 0.2, 0.4, 0.6
        family = rc.GaussianCopula(0.3)
        model = DynamicCopula(family, coefficients=(omega, alpha, beta), lags=5)
        path = model.filter(sample).path

        averaged = _moving_average(_forcing_values(sample, "normal-product", None), 5)
        link = model.link
        previous = 0.3
        for t in range(20):
            # On (-1, 1) the standardisation is the identity, so this is
            # literally Patton's equation.
            expected = float(link(omega + beta * previous + alpha * averaged[t]))
            assert path[t] == pytest.approx(expected)
            previous = expected

    def test_gas_score_is_the_derivative_of_the_log_density(self, sample: np.ndarray) -> None:
        model = DynamicCopula(rc.GaussianCopula(0.4), coefficients=(0, 0, 0), driver="gas")
        row, linked = sample[0], 0.5

        def value(f: float) -> float:
            return model._logpdf_at(row, float(model.link(f)))

        step = 1e-5
        reference = (value(linked + step) - value(linked - step)) / (2 * step)
        assert model._score(row, linked) == pytest.approx(reference, rel=1e-4)

    def test_gas_score_vanishes_where_the_link_saturates(self, sample: np.ndarray) -> None:
        # The chain rule through the link is what keeps the recursion inside the
        # domain: at the boundary the update has nothing to push against.
        model = DynamicCopula(rc.GaussianCopula(0.4), coefficients=(0, 0, 0), driver="gas")
        assert abs(model._score(sample[0], 60.0)) < 1e-6

    def test_rejects_the_wrong_shape(self) -> None:
        model = DynamicCopula(rc.GaussianCopula(0.5), coefficients=(0, 0, 0))
        with pytest.raises(ValueError, match=r"\(n, 2\)"):
            model.filter(np.random.default_rng(0).uniform(size=(10, 3)))


class TestSimulate:
    @pytest.mark.parametrize("driver", ["patton", "gas"])
    def test_shape_and_support(self, driver: str) -> None:
        model = DynamicCopula(rc.GaussianCopula(0.0), coefficients=(0.1, 0.2, 0.85), driver=driver)
        u = model.simulate(300, random_state=0)
        assert u.shape == (300, 2)
        assert np.all((u > 0) & (u < 1))

    def test_reproducible(self) -> None:
        model = DynamicCopula(rc.GaussianCopula(0.0), coefficients=(0.1, 0.2, 0.85))
        first = model.simulate(100, random_state=3)
        second = model.simulate(100, random_state=3)
        np.testing.assert_array_equal(first, second)

    def test_a_persistent_recursion_produces_dependence_that_moves(self) -> None:
        model = DynamicCopula(rc.GaussianCopula(0.0), coefficients=(0.3, 1.0, 0.8))
        u = model.simulate(3000, random_state=0)
        path = model.filter(u).path
        # Not a constant copula in disguise: the parameter really does travel.
        assert path.std() > 0.1
        assert path.max() - path.min() > 0.5

    def test_burn_in_removes_the_starting_value(self) -> None:
        # With a long burn-in the returned path should not remember where
        # family.params started.
        low = DynamicCopula(rc.GaussianCopula(-0.8), coefficients=(0.3, 1.0, 0.8))
        high = DynamicCopula(rc.GaussianCopula(0.8), coefficients=(0.3, 1.0, 0.8))
        a = low.simulate(500, random_state=5, burn_in=500)
        b = high.simulate(500, random_state=5, burn_in=500)
        assert abs(rc.cor_kendall(a)[0, 1] - rc.cor_kendall(b)[0, 1]) < 0.1


@pytest.fixture(scope="module")
def dynamic_fit() -> tuple[DynamicFitResult, np.ndarray]:
    truth = DynamicCopula(rc.GaussianCopula(0.0), coefficients=(0.30, 1.0, 0.80))
    u = truth.simulate(1500, random_state=0)
    return fit_dynamic(u, rc.GaussianCopula(0.0)), truth.filter(u).path


@pytest.fixture(scope="module")
def fitted_dcc() -> DccResult:
    u = rc.GaussianCopula(0.5, dim=3, dispstr="ex").rvs(500, random_state=0)
    return fit_dcc(u)


class TestFit:
    def test_recovers_the_path(self, dynamic_fit: tuple[DynamicFitResult, np.ndarray]) -> None:
        result, truth = dynamic_fit
        assert np.corrcoef(result.path, truth)[0, 1] > 0.9

    def test_beats_the_constant_model(
        self, dynamic_fit: tuple[DynamicFitResult, np.ndarray]
    ) -> None:
        result, _ = dynamic_fit
        assert result.loglik > result.constant_loglik

    def test_information_criteria(self, dynamic_fit: tuple[DynamicFitResult, np.ndarray]) -> None:
        result, _ = dynamic_fit
        assert result.n_params == 3
        assert result.aic == pytest.approx(6 - 2 * result.loglik)
        assert result.bic == pytest.approx(3 * np.log(result.n_obs) - 2 * result.loglik)
        assert result.bic > result.aic  # n > e^2, so BIC penalises harder

    def test_summary_reports_the_pieces(
        self, dynamic_fit: tuple[DynamicFitResult, np.ndarray]
    ) -> None:
        text = dynamic_fit[0].summary()
        for expected in ("omega", "alpha", "beta", "log-likelihood", "LR vs constant"):
            assert expected in text

    def test_constancy_test_is_a_valid_pair(
        self, dynamic_fit: tuple[DynamicFitResult, np.ndarray]
    ) -> None:
        statistic, pvalue = dynamic_fit[0].constancy_test()
        assert statistic >= 0
        assert 0.0 <= pvalue <= 1.0

    def test_constant_data_gives_a_nearly_flat_path(self) -> None:
        u = rc.GaussianCopula(0.6).rvs(1500, random_state=1)
        result = fit_dynamic(u, rc.GaussianCopula(0.0))
        assert result.path.std() < 0.15
        assert abs(result.path.mean() - 0.6) < 0.1

    def test_the_optimum_is_at_least_as_good_as_the_truth(self) -> None:
        truth = DynamicCopula(rc.ClaytonCopula(1.0), coefficients=(-1.2, -1.5, 0.85))
        u = truth.simulate(1200, random_state=0)
        result = fit_dynamic(u, rc.ClaytonCopula(1.0))
        # A maximiser that returns less than the generating coefficients has not
        # maximised anything.
        assert result.loglik >= truth.loglik(u) - 1e-6

    @pytest.mark.slow
    def test_gas_driver_recovers_a_path(self) -> None:
        truth = DynamicCopula(rc.GaussianCopula(0.0), coefficients=(0.10, 0.15, 0.95), driver="gas")
        u = truth.simulate(1200, random_state=0)
        result = fit_dynamic(u, rc.GaussianCopula(0.0), driver="gas")
        assert np.corrcoef(result.path, truth.filter(u).path)[0, 1] > 0.7
        assert result.loglik > result.constant_loglik

    def test_accepts_raw_data_by_converting_it(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=400)
        result = fit_dynamic(x, rc.GaussianCopula(0.0))
        assert np.isfinite(result.loglik)

    def test_explicit_start_is_used(self) -> None:
        u = rc.GaussianCopula(0.5).rvs(300, random_state=0)
        result = fit_dynamic(u, rc.GaussianCopula(0.0), start=(0.5, 0.1, 0.5))
        assert np.all(np.isfinite(result.coefficients))

    def test_rejects_the_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match=r"\(n, 2\)"):
            fit_dynamic(np.random.default_rng(0).uniform(size=(20, 3)), rc.GaussianCopula(0.0))


class TestForecast:
    def test_keys_and_shapes(self) -> None:
        model = DynamicCopula(rc.GaussianCopula(0.4), coefficients=(0.2, 0.3, 0.8))
        u = model.simulate(300, random_state=0)
        ahead = model.forecast(u, horizon=6, draws=150, random_state=1)
        assert sorted(ahead) == ["lower", "mean", "median", "upper"]
        for values in ahead.values():
            assert values.shape == (6,)

    def test_the_interval_brackets_the_centre(self) -> None:
        model = DynamicCopula(rc.GaussianCopula(0.4), coefficients=(0.2, 0.3, 0.8))
        u = model.simulate(300, random_state=0)
        ahead = model.forecast(u, horizon=5, draws=300, random_state=1)
        assert np.all(ahead["lower"] <= ahead["median"])
        assert np.all(ahead["median"] <= ahead["upper"])

    def test_stays_inside_the_parameter_range(self) -> None:
        model = DynamicCopula(rc.ClaytonCopula(2.0), coefficients=(0.3, -1.0, 0.9))
        u = model.simulate(200, random_state=0)
        ahead = model.forecast(u, horizon=4, draws=100, random_state=1)
        assert np.all(ahead["lower"] > model.link.lower)
        assert np.all(ahead["upper"] < model.link.upper)

    def test_uncertainty_widens_with_the_horizon(self) -> None:
        model = DynamicCopula(rc.GaussianCopula(0.2), coefficients=(0.3, 1.0, 0.8))
        u = model.simulate(400, random_state=0)
        ahead = model.forecast(u, horizon=10, draws=400, random_state=2)
        width = ahead["upper"] - ahead["lower"]
        assert width[-1] > width[0]

    def test_rejects_a_zero_horizon(self) -> None:
        model = DynamicCopula(rc.GaussianCopula(0.4), coefficients=(0.2, 0.3, 0.8))
        with pytest.raises(ValueError, match="at least 1"):
            model.forecast(model.simulate(50, random_state=0), horizon=0)


class TestDcc:
    def test_shape_and_validity(self, fitted_dcc: DccResult) -> None:
        assert fitted_dcc.correlations.shape == (500, 3, 3)
        for matrix in fitted_dcc.correlations[::50]:
            np.testing.assert_allclose(np.diag(matrix), 1.0)
            np.testing.assert_allclose(matrix, matrix.T)
            assert np.all(linalg.eigvalsh(matrix) > 0)

    def test_coefficients_are_admissible(self, fitted_dcc: DccResult) -> None:
        assert fitted_dcc.a >= 0
        assert fitted_dcc.b >= 0
        assert fitted_dcc.persistence < 1.0

    def test_constant_data_gives_a_stable_correlation(self, fitted_dcc: DccResult) -> None:
        path = fitted_dcc.pair(0, 1)
        assert abs(path.mean() - 0.5) < 0.12

    def test_likelihood_matches_the_gaussian_copula(self) -> None:
        # _dcc_loglik is written out by hand for speed. This is the only place
        # it can disagree with the copula it claims to be.
        rng = np.random.default_rng(0)
        u = rc.GaussianCopula(0.4, dim=3, dispstr="ex").rvs(200, random_state=0)
        z = stats.norm.ppf(u)
        matrix = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]])
        correlations = np.broadcast_to(matrix, (200, 3, 3)).copy()
        direct = _dcc_loglik(z, correlations, None)
        reference = float(np.sum(rc.GaussianCopula(rc.P2p(matrix), dim=3, dispstr="un").logpdf(u)))
        assert direct == pytest.approx(reference, rel=1e-9)
        del rng

    def test_likelihood_matches_the_student_copula(self) -> None:
        df = 6.0
        u = rc.GaussianCopula(0.4, dim=3, dispstr="ex").rvs(200, random_state=1)
        z = stats.t(df).ppf(u)
        matrix = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]])
        correlations = np.broadcast_to(matrix, (200, 3, 3)).copy()
        direct = _dcc_loglik(np.asarray(z), correlations, df)
        reference = float(
            np.sum(rc.StudentCopula(rc.P2p(matrix), df=df, dim=3, dispstr="un").logpdf(u))
        )
        assert direct == pytest.approx(reference, rel=1e-9)

    def test_zero_coefficients_give_the_unconditional_matrix(self) -> None:
        u = rc.GaussianCopula(0.5, dim=3, dispstr="ex").rvs(200, random_state=0)
        z = np.asarray(stats.norm.ppf(u))
        target = np.corrcoef(z, rowvar=False)
        correlations = _dcc_filter(z, target, 0.0, 0.0)
        for matrix in correlations[::40]:
            np.testing.assert_allclose(matrix, target, atol=1e-12)

    def test_correlation_moves_when_the_data_does(self) -> None:
        # First half independent, second half strongly dependent: the filtered
        # correlation has to notice.
        early = rc.IndependenceCopula(2).rvs(600, random_state=0)
        late = rc.GaussianCopula(0.85).rvs(600, random_state=1)
        result = fit_dcc(np.vstack([early, late]))
        path = result.pair(0, 1)
        assert path[500:600].mean() < path[1100:1200].mean() - 0.2

    def test_student_option(self) -> None:
        u = rc.StudentCopula(0.5, df=5.0, dim=3, dispstr="ex").rvs(400, random_state=0)
        result = fit_dcc(u, df=5.0)
        assert result.df == 5.0
        assert np.isfinite(result.loglik)
        assert "t copula" in result.summary()

    def test_copulas_are_built_on_request(self, fitted_dcc: DccResult) -> None:
        copulas = fitted_dcc.copulas()
        assert len(copulas) == fitted_dcc.n_obs
        assert all(c.dim == 3 for c in copulas[::100])

    def test_summary_reports_the_pieces(self, fitted_dcc: DccResult) -> None:
        text = fitted_dcc.summary()
        for expected in ("news impact", "persistence", "log-lik"):
            assert expected in text

    def test_singular_correlation_gives_minus_infinity(self) -> None:
        z = np.asarray(stats.norm.ppf(rc.GaussianCopula(0.5).rvs(20, random_state=0)))
        singular = np.broadcast_to(np.ones((2, 2)), (20, 2, 2)).copy()
        assert _dcc_loglik(z, singular, None) == -np.inf

    def test_rejects_a_univariate_input(self) -> None:
        with pytest.raises(ValueError, match=r"d >= 2"):
            fit_dcc(np.random.default_rng(0).uniform(size=(50, 1)))

    def test_accepts_raw_data(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.multivariate_normal(np.zeros(3), np.eye(3), size=200)
        result = fit_dcc(x)
        assert np.isfinite(result.loglik)
