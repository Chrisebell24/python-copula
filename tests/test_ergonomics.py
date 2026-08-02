"""Tests for :mod:`rcopula.serialize` and :mod:`rcopula.bootstrap`.

For serialization the bar is **exact**, not close: a round-tripped copula must
return bit-identical values. "Close enough" would mean a model checked into
version control prices a book differently next year, which is the one thing the
format exists to prevent. So the assertions are ``array_equal``, not
``allclose``.

For the bootstrap the bar is **coverage**. An interval that does not contain the
truth 95% of the time is not a 95% interval, however elegantly it was derived,
so the main test measures that directly over repeated samples rather than
checking one interval looks plausible.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import rcopula as rc
from rcopula.bootstrap import bootstrap, bootstrap_fit, bootstrap_measure
from rcopula.serialize import SCHEMA_VERSION, from_dict, from_json, to_dict, to_json

#: One of everything, including the awkward corners: a fixed parameter, an
#: unstructured correlation matrix, a Toeplitz one, a fixed-df t copula, both
#: Frechet bounds, and structural constructions nested two deep.
ROUND_TRIP = [
    rc.IndependenceCopula(3),
    rc.GaussianCopula(0.5),
    rc.GaussianCopula([0.1, 0.2, 0.3], dim=3, dispstr="un"),
    rc.GaussianCopula([0.3, 0.4], dim=3, dispstr="toep"),
    rc.StudentCopula(0.5, df=4.0),
    rc.StudentCopula(0.5, df=4.0, df_fixed=True),
    rc.ClaytonCopula(2.5, dim=3),
    rc.ClaytonCopula(-0.4),
    rc.FrankCopula(-3.0),
    rc.JoeCopula(2.0),
    rc.AMHCopula(0.7),
    rc.PlackettCopula(4.0),
    rc.FGMCopula(0.5),
    rc.GalambosCopula(1.0),
    rc.HuslerReissCopula(1.5),
    rc.TawnCopula(0.6),
    rc.TEVCopula(0.5, df=4.0),
    rc.MarshallOlkinCopula([0.4, 0.7]),
    rc.FrechetUpperCopula(2),
    rc.FrechetLowerCopula(2),
    rc.RotatedCopula(rc.ClaytonCopula(2.0), 90),
    rc.survival(rc.GumbelCopula(2.0)),
    rc.KhoudrajiCopula(
        rc.IndependenceCopula(2), rc.RotatedCopula(rc.GumbelCopula(3.0), 180), shapes=(0.4, 0.95)
    ),
    rc.MixtureCopula([rc.ClaytonCopula(3.0), rc.GumbelCopula(2.0)], weights=[0.4, 0.6]),
    rc.ClaytonCopula(2.0).fix_params([False]),
]


def _points(dim: int) -> np.ndarray:
    grid = np.linspace(0.07, 0.93, 7)
    return np.column_stack([np.roll(grid, j) for j in range(dim)])


class TestSerializeRoundTrip:
    @pytest.mark.parametrize("copula", ROUND_TRIP, ids=lambda c: c.describe()[:36])
    def test_the_cdf_is_bit_identical(self, copula: rc.Copula) -> None:
        reloaded = from_json(to_json(copula))
        points = _points(copula.dim)
        assert np.array_equal(np.asarray(copula.cdf(points)), np.asarray(reloaded.cdf(points)))

    @pytest.mark.parametrize("copula", ROUND_TRIP, ids=lambda c: c.describe()[:36])
    def test_the_description_survives(self, copula: rc.Copula) -> None:
        assert from_json(to_json(copula)).describe() == copula.describe()

    @pytest.mark.parametrize("copula", ROUND_TRIP, ids=lambda c: c.describe()[:36])
    def test_the_free_parameter_mask_survives(self, copula: rc.Copula) -> None:
        # A copula fitted with one parameter held fixed must come back fixed, or
        # a later refit would silently free it.
        assert np.array_equal(np.asarray(from_json(to_json(copula)).free), np.asarray(copula.free))

    @pytest.mark.parametrize("copula", ROUND_TRIP, ids=lambda c: c.describe()[:36])
    def test_the_parameters_are_exact(self, copula: rc.Copula) -> None:
        assert np.array_equal(
            np.asarray(from_json(to_json(copula)).params), np.asarray(copula.params)
        )

    def test_a_parameter_with_full_precision_survives(self) -> None:
        # Python's shortest round-tripping repr is what JSON writes, so this is
        # exact rather than merely 17 digits.
        theta = 2.3456789012345678
        assert from_json(to_json(rc.ClaytonCopula(theta))).params[0] == theta

    def test_the_density_is_bit_identical(self) -> None:
        original = rc.ClaytonCopula(2.3456789012345678, dim=4)
        u = original.rvs(200, random_state=0)
        assert np.array_equal(
            np.asarray(original.logpdf(u)), np.asarray(from_json(to_json(original)).logpdf(u))
        )

    def test_equality_holds_for_plain_families(self) -> None:
        for copula in (rc.ClaytonCopula(2.5, dim=3), rc.JoeCopula(3.0), rc.GaussianCopula(0.4)):
            assert from_json(to_json(copula)) == copula


class TestSerializeStructures:
    def test_a_vine_round_trips(self) -> None:
        data = rc.GaussianCopula(0.5, dim=4, dispstr="ex").rvs(400, random_state=0)
        vine = rc.fit_vine(data, structure="C")
        reloaded = from_json(to_json(vine))
        u = vine.rvs(50, random_state=1)
        assert np.array_equal(np.asarray(vine.logpdf(u)), np.asarray(reloaded.logpdf(u)))
        assert reloaded.structure == vine.structure
        assert list(reloaded.order) == list(vine.order)

    def test_a_vine_keeps_its_per_edge_families(self) -> None:
        # The whole point of a vine is a different family on every edge; a
        # format that flattened them would be useless.
        data = rc.GaussianCopula(0.5, dim=4, dispstr="ex").rvs(400, random_state=0)
        vine = rc.fit_vine(data, structure="C")
        reloaded = from_json(to_json(vine))
        for original_tree, new_tree in zip(vine.pair_copulas, reloaded.pair_copulas, strict=True):
            for original, new in zip(original_tree, new_tree, strict=True):
                assert type(new) is type(original)
                assert np.array_equal(np.asarray(new.params), np.asarray(original.params))

    def test_a_nested_archimedean_round_trips(self) -> None:
        tree = rc.NestedArchimedean(
            rc.ClaytonCopula(1.2),
            components=[0],
            children=[rc.NestedArchimedean(rc.ClaytonCopula(3.0), components=[1, 2, 3])],
        )
        u = tree.rvs(300, random_state=0)
        reloaded = from_json(to_json(tree))
        assert reloaded.describe() == tree.describe()
        assert np.array_equal(np.asarray(tree.cdf(u)), np.asarray(reloaded.cdf(u)))

    def test_nesting_is_preserved_three_deep(self) -> None:
        deep = rc.KhoudrajiCopula(
            rc.MixtureCopula(
                [rc.ClaytonCopula(2.0), rc.survival(rc.GumbelCopula(3.0))], weights=[0.3, 0.7]
            ),
            rc.RotatedCopula(rc.FrankCopula(5.0), 90),
            shapes=(0.5, 0.9),
        )
        reloaded = from_json(to_json(deep))
        points = _points(2)
        assert np.array_equal(np.asarray(deep.cdf(points)), np.asarray(reloaded.cdf(points)))


class TestSerializeFormat:
    def test_the_document_is_readable_json(self) -> None:
        document = json.loads(to_json(rc.GumbelCopula(2.0)))
        assert document["schema"] == SCHEMA_VERSION
        assert document["copula"]["kind"] == "GumbelCopula"
        assert document["copula"]["params"] == [2.0]
        assert "rcopula" in document

    def test_to_dict_and_from_dict_agree_with_the_json_pair(self) -> None:
        copula = rc.FrankCopula(4.0)
        assert from_dict(to_dict(copula)).describe() == from_json(to_json(copula)).describe()

    def test_a_future_schema_is_refused(self) -> None:
        document = to_dict(rc.ClaytonCopula(2.0))
        document["schema"] = SCHEMA_VERSION + 1
        with pytest.raises(ValueError, match="Upgrade"):
            from_dict(document)

    def test_a_missing_schema_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no 'schema'"):
            from_dict({"copula": {"kind": "ClaytonCopula"}})

    def test_a_missing_copula_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no 'copula'"):
            from_dict({"schema": SCHEMA_VERSION})

    def test_an_unknown_family_is_refused_by_name(self) -> None:
        document = to_dict(rc.ClaytonCopula(2.0))
        document["copula"]["kind"] = "InventedCopula"
        with pytest.raises(ValueError, match="InventedCopula"):
            from_dict(document)

    def test_a_node_without_a_kind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="'kind'"):
            from_dict({"schema": SCHEMA_VERSION, "copula": {"params": [2.0]}})

    def test_the_empirical_copula_is_refused_with_a_reason(self) -> None:
        data = rc.GaussianCopula(0.5).rvs(50, random_state=0)
        with pytest.raises(TypeError, match="licence"):
            to_json(rc.EmpiricalCopula(data))

    def test_compact_form_has_no_newlines(self) -> None:
        assert "\n" not in to_json(rc.ClaytonCopula(2.0), indent=None)


class TestBootstrapCoverage:
    """The only test of a confidence interval that means anything."""

    @pytest.mark.slow
    @pytest.mark.parametrize("method", ["bca", "percentile", "basic"])
    def test_a_95_percent_interval_covers_95_percent_of_the_time(self, method: str) -> None:
        truth = rc.ClaytonCopula(2.0).tau()
        covered = 0
        trials = 120
        for seed in range(trials):
            u = rc.ClaytonCopula(2.0).rvs(200, random_state=seed)
            lower, upper = bootstrap_measure(
                u, "tau", n_resamples=199, method=method, random_state=seed
            ).confidence_interval
            covered += bool(lower <= truth <= upper)
        rate = covered / trials
        # Binomial standard error at p = 0.95, n = 120 is 0.020, so +/- 0.07 is
        # three and a half of them -- tight enough to catch an interval that is
        # systematically too narrow, loose enough not to flake.
        assert 0.88 <= rate <= 1.0, f"{method} covered {rate:.1%}"

    @pytest.mark.slow
    def test_bca_is_not_wider_than_it_needs_to_be(self) -> None:
        # Coverage alone is satisfied by (-1, 1). The interval must also be
        # comparable in width to the asymptotic one where the asymptotics apply.
        u = rc.ClaytonCopula(2.0).rvs(1000, random_state=0)
        result = bootstrap_measure(u, "tau", n_resamples=499, random_state=0)
        lower, upper = result.confidence_interval
        # Asymptotic standard error of Kendall's tau is about sqrt(4/(9n)) times
        # a factor near 1; the interval should be a few tenths wide at most.
        assert (upper - lower) < 0.15
        assert result.standard_error < 0.05


class TestBootstrapBehaviour:
    def test_the_estimate_is_the_statistic_on_the_original_data(self) -> None:
        u = rc.GumbelCopula(2.0).rvs(300, random_state=0)
        result = bootstrap_measure(u, "tau", n_resamples=99, random_state=0)
        assert result.estimate == pytest.approx(float(rc.cor_kendall(u)[0, 1]))

    def test_the_interval_brackets_the_estimate(self) -> None:
        u = rc.ClaytonCopula(2.0).rvs(300, random_state=0)
        for method in ("bca", "percentile", "basic"):
            result = bootstrap_measure(u, "tau", n_resamples=199, method=method, random_state=0)
            lower, upper = result.confidence_interval
            assert lower <= result.estimate <= upper

    def test_a_wider_level_gives_a_wider_interval(self) -> None:
        u = rc.ClaytonCopula(2.0).rvs(300, random_state=0)
        narrow = bootstrap_measure(u, "tau", n_resamples=299, level=0.80, random_state=0)
        wide = bootstrap_measure(u, "tau", n_resamples=299, level=0.99, random_state=0)
        assert (wide.confidence_interval[1] - wide.confidence_interval[0]) > (
            narrow.confidence_interval[1] - narrow.confidence_interval[0]
        )

    def test_more_data_gives_a_narrower_interval(self) -> None:
        widths = []
        for n in (100, 400, 1600):
            u = rc.ClaytonCopula(2.0).rvs(n, random_state=0)
            result = bootstrap_measure(u, "tau", n_resamples=199, random_state=0)
            widths.append(result.confidence_interval[1] - result.confidence_interval[0])
        assert widths[0] > widths[1] > widths[2]
        # And it narrows at roughly the root-n rate: quadrupling n should nearly
        # halve the width.
        assert 1.4 < widths[0] / widths[2] / 2.0 < 3.0

    def test_resampling_keeps_whole_rows(self) -> None:
        # If rows were broken apart, the dependence would vanish and the
        # replicates would centre on zero rather than on the estimate.
        u = rc.ClaytonCopula(6.0).rvs(400, random_state=0)
        result = bootstrap_measure(u, "tau", n_resamples=299, random_state=0)
        assert float(np.mean(result.replicates)) > 0.6
        assert abs(float(np.mean(result.replicates)) - result.estimate) < 0.02

    def test_it_is_reproducible(self) -> None:
        u = rc.ClaytonCopula(2.0).rvs(200, random_state=0)
        first = bootstrap_measure(u, "tau", n_resamples=99, random_state=7)
        second = bootstrap_measure(u, "tau", n_resamples=99, random_state=7)
        assert np.array_equal(first.replicates, second.replicates)

    @pytest.mark.parametrize("measure", ["tau", "rho", "beta", "lambda_upper", "lambda_lower"])
    def test_every_named_measure_works(self, measure: str) -> None:
        u = rc.ClaytonCopula(2.0).rvs(400, random_state=0)
        result = bootstrap_measure(u, measure, n_resamples=99, random_state=0)
        assert np.isfinite(result.estimate)
        assert result.confidence_interval[0] <= result.confidence_interval[1]

    def test_tail_dependence_intervals_are_asymmetric(self) -> None:
        # The case that motivates the whole module: a ratio of small counts,
        # bounded below at zero, so the sampling distribution is skewed and a
        # symmetric asymptotic interval would misrepresent it.
        u = rc.ClaytonCopula(2.0).rvs(500, random_state=0)
        result = bootstrap_measure(u, "lambda_lower", n_resamples=499, random_state=0)
        lower, upper = result.confidence_interval
        assert lower >= 0.0
        below = result.estimate - lower
        above = upper - result.estimate
        assert abs(below - above) > 0.01 * result.estimate

    def test_summary_reports_the_pieces(self) -> None:
        u = rc.ClaytonCopula(2.0).rvs(200, random_state=0)
        text = bootstrap_measure(u, "tau", n_resamples=99, random_state=0).summary()
        for expected in ("estimate", "SE", "bias", "95% lower", "95% upper"):
            assert expected in text

    def test_bias_is_reported(self) -> None:
        u = rc.ClaytonCopula(2.0).rvs(300, random_state=0)
        result = bootstrap_measure(u, "tau", n_resamples=199, random_state=0)
        assert abs(float(result.bias)) < 0.05


class TestBootstrapFit:
    def test_it_covers_the_generating_parameter(self) -> None:
        u = rc.ClaytonCopula(2.0).rvs(400, random_state=0)
        result = bootstrap_fit(
            u, rc.ClaytonCopula(1.0), n_resamples=99, method="percentile", random_state=0
        )
        lower, upper = result.confidence_interval
        assert float(np.ravel(lower)[0]) < 2.0 < float(np.ravel(upper)[0])

    def test_it_agrees_roughly_with_the_asymptotic_standard_error(self) -> None:
        # Where the asymptotics apply the two should be close; a bootstrap SE
        # twice the asymptotic one would mean one of them is wrong.
        u = rc.ClaytonCopula(2.0).rvs(800, random_state=0)
        asymptotic = float(np.ravel(rc.fit(rc.ClaytonCopula(1.0), u, method="mpl").bse)[0])
        boot = bootstrap_fit(
            u, rc.ClaytonCopula(1.0), n_resamples=199, method="percentile", random_state=0
        )
        ratio = float(np.ravel(boot.standard_error)[0]) / asymptotic
        assert 0.7 < ratio < 1.4

    def test_a_two_parameter_copula_gives_two_intervals(self) -> None:
        u = rc.StudentCopula(0.5, df=5.0).rvs(400, random_state=0)
        result = bootstrap_fit(
            u, rc.StudentCopula(0.0, df=8.0), n_resamples=59, method="percentile", random_state=0
        )
        assert np.asarray(result.estimate).size == 2
        lower, upper = result.confidence_interval
        assert np.asarray(lower).size == 2
        assert np.all(np.asarray(lower) <= np.asarray(upper))


class TestBootstrapValidation:
    def test_rejects_a_single_observation(self) -> None:
        with pytest.raises(ValueError, match="at least 2 observations"):
            bootstrap(np.array([[0.5, 0.5]]), lambda d: float(d.mean()))

    def test_rejects_a_level_outside_the_unit_interval(self) -> None:
        u = rc.ClaytonCopula(2.0).rvs(50, random_state=0)
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            bootstrap(u, lambda d: float(d.mean()), level=1.5)

    def test_rejects_too_few_resamples(self) -> None:
        u = rc.ClaytonCopula(2.0).rvs(50, random_state=0)
        with pytest.raises(ValueError, match="at least 2"):
            bootstrap(u, lambda d: float(d.mean()), n_resamples=1)

    def test_rejects_an_unknown_method(self) -> None:
        u = rc.ClaytonCopula(2.0).rvs(50, random_state=0)
        with pytest.raises(ValueError, match="bca, percentile or basic"):
            bootstrap(u, lambda d: float(d.mean()), method="jackknife")

    def test_rejects_an_unknown_measure(self) -> None:
        u = rc.ClaytonCopula(2.0).rvs(50, random_state=0)
        with pytest.raises(ValueError, match="unknown measure"):
            bootstrap_measure(u, "gini")

    def test_rejects_a_non_bivariate_measure_call(self) -> None:
        u = rc.ClaytonCopula(2.0, dim=3).rvs(50, random_state=0)
        with pytest.raises(ValueError, match="2 columns"):
            bootstrap_measure(u, "tau")

    def test_a_statistic_that_always_fails_is_reported(self) -> None:
        u = rc.ClaytonCopula(2.0).rvs(50, random_state=0)
        with pytest.raises(ValueError, match="failed on the original data"):
            bootstrap(u, _always_raises)

    def test_occasional_failures_are_tolerated(self) -> None:
        # A resample can be degenerate and an estimator is entitled to refuse
        # it; that should cost one replicate, not the run.
        u = rc.ClaytonCopula(2.0).rvs(200, random_state=0)
        result = bootstrap(u, _fails_rarely, n_resamples=199, method="percentile", random_state=0)
        assert result.replicates.shape[0] <= 199
        assert np.isfinite(result.estimate)

    def test_too_many_failures_raises(self) -> None:
        u = rc.ClaytonCopula(2.0).rvs(200, random_state=0)
        with pytest.raises(RuntimeError, match="too many to be incidental"):
            bootstrap(u, _fails_often, n_resamples=99, random_state=0)


def _always_raises(x: np.ndarray) -> float:
    raise RuntimeError("no")


def _fails_rarely(x: np.ndarray) -> float:
    value = float(np.mean(x[:, 0]))
    if value > 0.53:
        raise RuntimeError("refused")
    return value


def _fails_often(x: np.ndarray) -> float:
    # Refuses anything with a repeated value, which every bootstrap resample has
    # and the original continuous sample does not. So the original succeeds and
    # essentially every resample fails, which is the case the guard is for.
    if np.unique(x[:, 0]).size < x.shape[0]:
        raise RuntimeError("refused")
    return float(np.mean(x[:, 0]))
