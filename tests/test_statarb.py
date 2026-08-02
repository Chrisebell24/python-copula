"""Tests for :mod:`rcopula.statarb`.

The selection rules are only worth having if they *disagree*, so several tests
construct data where a specific rule should win and check that it does -- a pair
that is linearly correlated but never jointly extreme, and a pair that is the
reverse. A module where every criterion picked the same candidates would be five
functions doing one job.

``multivariate_spearman`` is pinned at both ends (zero under independence, one
under comonotonicity, in every dimension) and against the bivariate coefficient
it generalises, which is the only external check available for it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import rcopula as rc
from rcopula.statarb import (
    PAIR_METHODS,
    PARTNER_METHODS,
    diagonal_distance,
    multivariate_spearman,
    select_pairs,
    select_partners,
    tail_concentration,
)


@pytest.fixture(scope="module")
def factor_panel() -> pd.DataFrame:
    """Three names on a common factor, three independent."""
    rng = np.random.default_rng(0)
    factor = rng.standard_normal(800)
    data = {f"F{k}": factor + 0.3 * rng.standard_normal(800) for k in range(1, 4)}
    data.update({f"N{k}": rng.standard_normal(800) for k in range(1, 4)})
    return pd.DataFrame(data)


class TestMultivariateSpearman:
    @pytest.mark.parametrize("rho", [-0.5, 0.0, 0.4, 0.85])
    def test_it_generalises_the_bivariate_coefficient(self, rho: float) -> None:
        u = rc.GaussianCopula(rho).rvs(40_000, random_state=0)
        assert multivariate_spearman(u) == pytest.approx(float(rc.cor_spearman(u)[0, 1]), abs=0.02)

    @pytest.mark.parametrize("dim", [2, 3, 4, 6])
    def test_zero_under_independence(self, dim: int) -> None:
        u = rc.IndependenceCopula(dim).rvs(40_000, random_state=0)
        assert abs(multivariate_spearman(u)) < 0.03

    @pytest.mark.parametrize("dim", [2, 3, 4, 6])
    def test_one_under_comonotonicity(self, dim: int) -> None:
        shared = np.random.default_rng(0).uniform(size=(20_000, 1))
        assert multivariate_spearman(np.tile(shared, (1, dim))) > 0.99

    def test_it_increases_with_dependence(self) -> None:
        values = [
            multivariate_spearman(
                rc.GaussianCopula(rho, dim=4, dispstr="ex").rvs(20_000, random_state=0)
            )
            for rho in (0.1, 0.4, 0.7, 0.9)
        ]
        assert np.all(np.diff(values) > 0)

    def test_it_ranks_the_data_itself(self) -> None:
        # Raw uniforms are in [0, 1] but are not ranks. Skipping the transform
        # for them costs about 0.02 on a comonotone quadruple, which is small
        # enough to pass for noise and large enough to reorder a shortlist.
        shared = np.random.default_rng(0).uniform(size=(20_000, 1))
        comonotone = np.tile(shared, (1, 4))
        assert multivariate_spearman(comonotone) > 0.99
        # And an arbitrary increasing marginal transform changes nothing.
        warped = np.column_stack([np.exp(comonotone[:, 0]), np.tan(comonotone[:, 1:] * 1.4)])
        assert multivariate_spearman(warped) == pytest.approx(
            multivariate_spearman(comonotone), abs=1e-9
        )

    def test_rejects_a_single_column(self) -> None:
        with pytest.raises(ValueError, match="at least 2 columns"):
            multivariate_spearman(np.random.default_rng(0).uniform(size=(50, 1)))


class TestGeometricAndExtremal:
    def test_diagonal_distance_is_zero_on_the_diagonal(self) -> None:
        shared = np.random.default_rng(0).uniform(size=(5000, 1))
        assert diagonal_distance(np.tile(shared, (1, 4))) < 1e-12

    def test_diagonal_distance_grows_as_dependence_falls(self) -> None:
        values = [
            diagonal_distance(rc.GaussianCopula(rho, dim=4, dispstr="ex").rvs(5000, random_state=0))
            for rho in (0.95, 0.7, 0.3, 0.0)
        ]
        assert np.all(np.diff(values) > 0)

    @pytest.mark.parametrize("dim", [2, 3, 4, 5])
    def test_tail_concentration_is_calibrated_in_every_dimension(self, dim: int) -> None:
        # The adaptive corner exists for this. At a fixed 5% corner and d = 4, a
        # 200,000-row sample expects 1.3 observations in it and the statistic
        # scatters between 0.4 and 1.2 on pure noise.
        values = [
            tail_concentration(rc.IndependenceCopula(dim).rvs(4000, random_state=seed))
            for seed in range(6)
        ]
        assert abs(float(np.mean(values)) - 1.0) < 0.3

    def test_a_corner_outside_the_unit_interval_is_refused(self) -> None:
        u = rc.IndependenceCopula(2).rvs(500, random_state=0)
        with pytest.raises(ValueError, match=r"\(0, 0.5\)"):
            tail_concentration(u, quantile=0.9)

    def test_tail_concentration_sees_what_correlation_does_not(self) -> None:
        # Clayton and Gumbel at the same Kendall tau: identical concordance,
        # opposite tails, and a large joint-tail concentration for both.
        for copula in (rc.ClaytonCopula.from_tau(0.5, dim=3), rc.GumbelCopula.from_tau(0.5, dim=3)):
            assert tail_concentration(copula.rvs(100_000, random_state=0)) > 5.0

    def test_it_separates_tail_behaviour_only_deep_in_the_corner(self) -> None:
        # At the same Kendall's tau, Clayton has lower tail dependence and
        # Gaussian has none. The statistic can see that -- but only at a corner
        # small enough that four dimensions could never supply the data.
        clayton = rc.ClaytonCopula.from_tau(0.5).rvs(200_000, random_state=0)
        gaussian = rc.GaussianCopula.from_tau(0.5).rvs(200_000, random_state=0)
        wide = tail_concentration(clayton, 0.30) / tail_concentration(gaussian, 0.30)
        deep = tail_concentration(clayton, 0.01) / tail_concentration(gaussian, 0.01)
        assert wide < 1.05  # indistinguishable
        assert deep > 1.3  # clearly separated

    def test_the_default_corner_cannot_separate_them_in_four_dimensions(self) -> None:
        """Pins the caveat, so it cannot be quietly forgotten.

        A four-dimensional corner deep enough to isolate the tail is empty at
        any realistic sample size, so the adaptive corner widens until it holds
        data -- and a wide corner measures joint co-movement, not tails. Anyone
        reaching for ``extremal`` expecting tail-driven selection should know
        that before they rely on it.
        """
        clayton = rc.ClaytonCopula.from_tau(0.5, dim=4).rvs(4000, random_state=0)
        gaussian = rc.GaussianCopula.from_tau(0.5, dim=4, dispstr="ex").rvs(4000, random_state=0)
        ratio = tail_concentration(clayton) / tail_concentration(gaussian)
        assert 0.9 < ratio < 1.1


class TestSelectPairs:
    @pytest.mark.parametrize("method", PAIR_METHODS)
    def test_every_method_finds_the_obvious_pair(
        self, method: str, factor_panel: pd.DataFrame
    ) -> None:
        best = select_pairs(factor_panel, method=method).iloc[0]
        assert best["first"].startswith("F")
        assert best["second"].startswith("F")

    @pytest.mark.parametrize("method", PAIR_METHODS)
    def test_larger_is_always_better(self, method: str, factor_panel: pd.DataFrame) -> None:
        # The distance and QQ rules are negated so that this holds for all six;
        # the ranking code depends on it.
        ranked = select_pairs(factor_panel, method=method)
        assert np.all(np.diff(ranked["score"].to_numpy()) <= 1e-12)
        assert ranked["rank"].tolist() == list(range(1, len(ranked) + 1))

    def test_every_pair_appears_exactly_once(self, factor_panel: pd.DataFrame) -> None:
        ranked = select_pairs(factor_panel, method="kendall")
        assert len(ranked) == 15  # 6 choose 2
        pairs = {frozenset((r["first"], r["second"])) for _, r in ranked.iterrows()}
        assert len(pairs) == 15

    def test_the_criteria_disagree(self) -> None:
        """The justification for having six of them.

        One pair whose cumulative *paths* track while their daily moves are
        noisy; another whose daily moves track closely while the paths drift
        apart. The level-based rules pick the first, the rank-based rules the
        second, and both are defensible answers to different questions.
        """
        rng = np.random.default_rng(3)
        n = 1500
        common = rng.standard_normal(n)
        a1 = common + 1.2 * rng.standard_normal(n)
        a2 = common + 1.2 * rng.standard_normal(n)
        drift = rng.standard_normal(n)
        b1 = drift + 0.25 * rng.standard_normal(n) + 0.04
        b2 = drift + 0.25 * rng.standard_normal(n) - 0.04
        frame = pd.DataFrame({"A1": a1, "A2": a2, "B1": b1, "B2": b2})
        level_based = {
            method: set(select_pairs(frame, method=method).iloc[0][["first", "second"]])
            for method in ("distance", "qq")
        }
        rank_based = {
            method: set(select_pairs(frame, method=method).iloc[0][["first", "second"]])
            for method in ("kendall", "spearman", "pearson")
        }
        assert all(chosen == {"A1", "A2"} for chosen in level_based.values())
        assert all(chosen == {"B1", "B2"} for chosen in rank_based.values())

    def test_top_truncates(self, factor_panel: pd.DataFrame) -> None:
        assert len(select_pairs(factor_panel, method="kendall", top=4)) == 4

    def test_it_accepts_a_bare_array(self, factor_panel: pd.DataFrame) -> None:
        ranked = select_pairs(factor_panel.to_numpy(), method="kendall", names=list("abcdef"))
        assert set(ranked.iloc[0][["first", "second"]]) <= set("abcdef")

    def test_the_method_travels_with_the_result(self, factor_panel: pd.DataFrame) -> None:
        assert select_pairs(factor_panel, method="qq").attrs["method"] == "qq"

    def test_rejects_an_unknown_method(self, factor_panel: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="method must be one of"):
            select_pairs(factor_panel, method="cointegration")

    def test_rejects_a_single_column(self) -> None:
        with pytest.raises(ValueError, match="at least 2 columns"):
            select_pairs(pd.DataFrame({"only": [1.0, 2.0, 3.0]}))


class TestSelectPartners:
    @pytest.mark.parametrize("method", PARTNER_METHODS)
    def test_every_method_recovers_a_planted_quadruple(self, method: str) -> None:
        rng = np.random.default_rng(0)
        factor = rng.standard_normal(1000)
        frame = pd.DataFrame(
            {
                "TGT": factor + 0.3 * rng.standard_normal(1000),
                **{f"P{k}": factor + 0.3 * rng.standard_normal(1000) for k in (1, 2, 3)},
                **{f"N{k}": rng.standard_normal(1000) for k in (1, 2, 3)},
            }
        )
        found = select_partners(frame, "TGT", method=method)
        assert sorted(found["partners"]) == ["P1", "P2", "P3"]
        assert found["target"] == "TGT"
        assert found["method"] == method

    def test_it_reports_how_much_it_actually_searched(self) -> None:
        # The pre-screening is the reason this terminates at all, so it must be
        # visible rather than implicit.
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(rng.standard_normal((400, 12)), columns=[f"c{k}" for k in range(12)])
        wide = select_partners(frame, "c0", n_candidates=11)
        narrow = select_partners(frame, "c0", n_candidates=5)
        assert wide["considered"] == 165  # 11 choose 3
        assert narrow["considered"] == 10  # 5 choose 3
        assert narrow["considered"] < wide["considered"]

    def test_the_extremal_rule_prefers_joint_tails(self) -> None:
        # Two candidate groups at similar rank correlation, one of which crashes
        # together. The extremal rule should split them.
        rng = np.random.default_rng(2)
        n = 4000
        tailed = rc.ClaytonCopula.from_tau(0.45, dim=4).rvs(n, random_state=0)
        smooth = rc.FrankCopula.from_tau(0.45, dim=3).rvs(n, random_state=1)
        frame = pd.DataFrame(
            {
                "TGT": tailed[:, 0],
                "C1": tailed[:, 1],
                "C2": tailed[:, 2],
                "C3": tailed[:, 3],
                "F1": smooth[:, 0],
                "F2": smooth[:, 1],
                "F3": smooth[:, 2],
            }
        )
        # Tie the Frank block to the target so it is a genuine competitor.
        frame["F1"] = 0.5 * frame["F1"] + 0.5 * frame["TGT"]
        found = select_partners(frame, "TGT", method="extremal")
        assert sorted(found["partners"]) == ["C1", "C2", "C3"]
        del rng

    def test_a_target_given_by_index(self) -> None:
        rng = np.random.default_rng(0)
        factor = rng.standard_normal(600)
        frame = pd.DataFrame(
            {
                "a": factor + 0.3 * rng.standard_normal(600),
                "b": factor + 0.3 * rng.standard_normal(600),
                "c": factor + 0.3 * rng.standard_normal(600),
                "d": factor + 0.3 * rng.standard_normal(600),
                "e": rng.standard_normal(600),
            }
        )
        assert select_partners(frame, 0)["target"] == "a"

    def test_n_partners_is_respected(self, factor_panel: pd.DataFrame) -> None:
        for count in (1, 2, 4):
            assert len(select_partners(factor_panel, "F1", n_partners=count)["partners"]) == count

    def test_rejects_an_unknown_method(self, factor_panel: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="method must be one of"):
            select_partners(factor_panel, "F1", method="mangold")

    def test_rejects_an_unknown_target(self, factor_panel: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="is not a column"):
            select_partners(factor_panel, "NOPE")

    def test_rejects_asking_for_more_partners_than_exist(self) -> None:
        values = np.random.default_rng(0).standard_normal((100, 3))
        frame = pd.DataFrame(values, columns=list("abc"))
        with pytest.raises(ValueError, match="besides the target"):
            select_partners(frame, "a", n_partners=5)

    def test_rejects_zero_partners(self, factor_panel: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            select_partners(factor_panel, "F1", n_partners=0)


class TestItComposesWithTheRestOfThePackage:
    def test_a_selected_pair_can_be_fitted_and_traded(self, factor_panel: pd.DataFrame) -> None:
        # The point of the module: its output is the next function's input.
        best = select_pairs(factor_panel, method="kendall").iloc[0]
        u = rc.pseudo_obs(factor_panel[[best["first"], best["second"]]])
        chosen = rc.select_copula(u, criterion="aic").best
        signal = rc.portfolio.pairs_signal(chosen, np.asarray(u))
        assert np.asarray(signal).shape[0] == len(factor_panel)

    def test_a_selected_quadruple_can_be_fitted_as_a_vine(self) -> None:
        rng = np.random.default_rng(0)
        factor = rng.standard_normal(800)
        frame = pd.DataFrame(
            {
                "TGT": factor + 0.3 * rng.standard_normal(800),
                **{f"P{k}": factor + 0.3 * rng.standard_normal(800) for k in (1, 2, 3)},
                **{f"N{k}": rng.standard_normal(800) for k in (1, 2)},
            }
        )
        found = select_partners(frame, "TGT", method="extended")
        block = frame[[found["target"], *found["partners"]]]
        vine = rc.fit_vine(rc.pseudo_obs(block), structure="D")
        assert vine.dim == 4
        assert np.all(np.isfinite(vine.logpdf(rc.pseudo_obs(block))))
