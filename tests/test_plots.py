"""Tests for the diagnostic plots.

A plot is hard to assert on, so these check the two things that can actually be
wrong: the **numbers behind the picture**, and the contract (returns axes, does
not show anything, refuses bad input). The drawing itself is exercised only for
absence of exceptions.

The numeric parts are worth checking carefully because they carry real content:
the tail concentration function must converge to the analytic tail-dependence
coefficients, and the Kendall plot's reference curve is an order-statistic
expectation whose exact mean is known.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import rcopula as rc
from rcopula.kendall import kendall_empirical
from rcopula.plots import (
    _concentration,
    _independence_order_statistics,
    contour,
    dependence_heatmap,
    kendall_plot,
    nested_tree,
    pickands_plot,
    scatter_matrix,
    surface,
    tail_concentration,
    vine_trees,
)
from rcopula.structural import KhoudrajiCopula, NestedArchimedean


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestTailConcentrationNumbers:
    """The curve must converge to the tail-dependence coefficients it depicts."""

    @pytest.mark.parametrize(
        "cop",
        [
            rc.ClaytonCopula.from_tau(0.5),
            rc.GumbelCopula.from_tau(0.5),
            rc.StudentCopula.from_tau(0.5, df=4.0),
            rc.FrankCopula.from_tau(0.5),
        ],
    )
    def test_the_branches_converge_to_lambda(self, cop: rc.Copula) -> None:
        q = np.array([1e-4])
        lower = _concentration(q, cop.cdf(np.column_stack([q, q])))[0]
        upper = _concentration(1 - q, cop.cdf(np.column_stack([1 - q, 1 - q])))[0]
        assert lower == pytest.approx(cop.lambda_().lower, abs=0.03)
        assert upper == pytest.approx(cop.lambda_().upper, abs=0.03)

    def test_the_gaussian_converges_far_too_slowly_to_be_reassuring(self) -> None:
        """The trap this plot is for, and why lambda = 0 is not the whole story.

        The Gaussian copula's tail-dependence coefficient is exactly zero, and
        its tail concentration reaches that limit logarithmically -- at
        tau = 0.5 it is still 0.27 at the 1-in-100 level and 0.10 at
        1-in-10,000. A Student-t at the same tau has settled to its limit by
        then. So "no tail dependence" describes behaviour at quantiles nobody
        observes, while at the levels a risk report actually uses the Gaussian
        looks substantially tail-dependent.
        """
        gaussian = rc.GaussianCopula.from_tau(0.5)
        student = rc.StudentCopula.from_tau(0.5, df=4.0)
        assert gaussian.lambda_().lower == 0.0

        levels = np.array([1e-2, 1e-4, 1e-8])
        curve = _concentration(levels, gaussian.cdf(np.column_stack([levels, levels])))
        assert curve[0] == pytest.approx(0.273, abs=0.01)
        assert curve[1] == pytest.approx(0.103, abs=0.01)
        assert curve[2] == pytest.approx(0.017, abs=0.01)

        # The t has essentially arrived by the 1-in-100 level; the Gaussian has
        # not arrived by 1-in-100-million.
        settled = _concentration(levels, student.cdf(np.column_stack([levels, levels])))
        assert settled[0] == pytest.approx(student.lambda_().lower, abs=0.04)
        assert curve[2] > 5 * gaussian.lambda_().lower + 0.01

    def test_it_separates_families_with_identical_tau(self) -> None:
        """The reason the plot exists: equal tau, opposite corners."""
        q = np.array([0.01, 0.99])
        clayton = _concentration(q, rc.ClaytonCopula.from_tau(0.5).cdf(np.column_stack([q, q])))
        gumbel = _concentration(q, rc.GumbelCopula.from_tau(0.5).cdf(np.column_stack([q, q])))
        assert clayton[0] > gumbel[0] + 0.4  # lower tail: Clayton dominates
        assert gumbel[1] > clayton[1] + 0.4  # upper tail: Gumbel does

    def test_it_is_bounded_and_meets_at_the_centre(self) -> None:
        cop = rc.GumbelCopula(2.0)
        q = np.linspace(0.001, 0.999, 400)
        values = _concentration(q, cop.cdf(np.column_stack([q, q])))
        assert np.all((values >= 0.0) & (values <= 1.0 + 1e-12))
        below = _concentration(np.array([0.5 - 1e-9]), cop.cdf([[0.5 - 1e-9, 0.5 - 1e-9]]))
        above = _concentration(np.array([0.5 + 1e-9]), cop.cdf([[0.5 + 1e-9, 0.5 + 1e-9]]))
        assert below[0] == pytest.approx(above[0], abs=1e-6)


class TestKendallPlotReference:
    """The reference curve is ``E[W_{i:n}]`` under independence."""

    @pytest.mark.parametrize("n", [10, 100, 1000, 2000])
    def test_its_mean_is_exactly_one_quarter(self, n: int) -> None:
        r"""``E[W] = int_0^1 -w log w dw = 1/4`` for every ``n``.

        An exact identity, and the sharpest available check on a quantity built
        from a binomial coefficient and an integral that diverge in opposite
        directions.
        """
        assert _independence_order_statistics(n).mean() == pytest.approx(0.25, abs=1e-9)

    @pytest.mark.parametrize("n", [10, 100, 1000, 2000, 5000])
    def test_it_stays_finite_and_ordered(self, n: int) -> None:
        """At n = 2000 the binomial alone is e^1403 and the integral e^-1400,
        so computing either separately gives inf * 0 = nan."""
        expected = _independence_order_statistics(n)
        assert np.all(np.isfinite(expected))
        assert np.all(np.diff(expected) >= -1e-12)
        assert np.all((expected >= 0.0) & (expected <= 1.0))

    @pytest.mark.parametrize(
        ("cop", "direction"),
        [
            (rc.IndependenceCopula(2), 0.0),
            (rc.ClaytonCopula(4.0), 1.0),
            (rc.GumbelCopula(3.0), 1.0),
            (rc.FrankCopula(-8.0), -1.0),
        ],
    )
    def test_the_bow_direction_tracks_the_dependence(
        self, cop: rc.Copula, direction: float
    ) -> None:
        """Above the diagonal for positive dependence, below for negative."""
        w = kendall_empirical(cop.rvs(2000, random_state=0))
        gap = float(np.mean(w - _independence_order_statistics(w.size)))
        if direction == 0.0:
            assert abs(gap) < 0.02
        else:
            assert np.sign(gap) == direction
            assert abs(gap) > 0.1

    def test_comonotonicity_reaches_the_maximum_gap(self) -> None:
        """``W_i = i/(n-1)`` has mean 1/2 against the reference's 1/4."""
        w = kendall_empirical(rc.FrechetUpperCopula(2).rvs(1000, random_state=0))
        gap = float(np.mean(w - _independence_order_statistics(w.size)))
        assert gap == pytest.approx(0.25, abs=0.01)


class TestPlotsRunAndReturnAxes:
    @pytest.mark.parametrize("kind", ["pdf", "logpdf", "cdf"])
    def test_contour(self, kind: str) -> None:
        ax = contour(rc.ClaytonCopula(3.0), kind=kind, n=20)
        assert ax.get_xlabel() == "u1"
        assert ax.collections or ax.get_children()

    def test_contour_accepts_an_existing_axes(self) -> None:
        _, ax = plt.subplots()
        assert contour(rc.GumbelCopula(2.0), n=20, ax=ax) is ax

    @pytest.mark.parametrize("kind", ["pdf", "cdf"])
    def test_surface(self, kind: str) -> None:
        ax = surface(rc.FrankCopula(5.0), kind=kind, n=15)
        assert ax.get_zlabel() == kind

    def test_scatter_matrix(self) -> None:
        u = rc.ClaytonCopula(3.0, dim=3).rvs(200, random_state=0)
        axes = scatter_matrix(u, names=["a", "b", "c"])
        assert axes.shape == (3, 3)
        assert axes[2][0].get_xlabel() == "a"
        assert axes[0][0].get_ylabel() == "a"

    def test_scatter_matrix_shows_tau_in_the_upper_triangle(self) -> None:
        u = rc.ClaytonCopula(3.0).rvs(200, random_state=0)
        axes = scatter_matrix(u)
        text = axes[0][1].texts[0].get_text()
        assert "tau" in text
        assert f"{rc.cor_kendall(u)[0, 1]:+.3f}" in text

    def test_tail_concentration_with_data_and_copulas(self) -> None:
        u = rc.ClaytonCopula.from_tau(0.5).rvs(500, random_state=0)
        ax = tail_concentration(u, [rc.ClaytonCopula.from_tau(0.5), rc.GumbelCopula.from_tau(0.5)])
        labels = [line.get_label() for line in ax.get_lines()]
        assert "empirical" in labels
        assert "Clayton" in labels and "Gumbel" in labels

    def test_tail_concentration_accepts_either_argument_alone(self) -> None:
        assert tail_concentration(copula=rc.GumbelCopula(2.0)) is not None
        assert tail_concentration(rc.GumbelCopula(2.0).rvs(200, random_state=0)) is not None

    def test_kendall_plot(self) -> None:
        u = rc.ClaytonCopula(3.0).rvs(300, random_state=0)
        ax = kendall_plot(u)
        assert ax.get_xlabel() == "expected under independence"
        assert len(ax.collections) == 1

    def test_pickands_plot(self) -> None:
        ax = pickands_plot([rc.GumbelCopula(2.0), rc.GalambosCopula(1.5)])
        labels = [line.get_label() for line in ax.get_lines()]
        assert "Gumbel" in labels and "Galambos" in labels

    def test_pickands_plot_accepts_a_khoudraji_copula(self) -> None:
        """Which is the interesting case -- an *asymmetric* Pickands function."""
        cop = KhoudrajiCopula(rc.IndependenceCopula(2), rc.GumbelCopula(3.0), [0.4, 0.95])
        ax = pickands_plot(cop)
        curve = ax.get_lines()[-1].get_ydata()
        assert abs(curve[50] - curve[-50]) > 0.01  # not symmetric about t = 1/2


class TestPlotsRejectBadInput:
    @pytest.mark.parametrize("fn", [contour, surface])
    def test_they_are_bivariate(self, fn) -> None:
        with pytest.raises(ValueError, match="bivariate"):
            fn(rc.ClaytonCopula(2.0, dim=3))

    def test_contour_rejects_an_unknown_kind(self) -> None:
        with pytest.raises(ValueError, match="kind must be"):
            contour(rc.ClaytonCopula(2.0), kind="survival")

    def test_scatter_matrix_checks_the_names(self) -> None:
        u = rc.ClaytonCopula(2.0, dim=3).rvs(50, random_state=0)
        with pytest.raises(ValueError, match="names for"):
            scatter_matrix(u, names=["a", "b"])

    def test_tail_concentration_needs_something_to_plot(self) -> None:
        with pytest.raises(ValueError, match="give data, a copula, or both"):
            tail_concentration()

    def test_tail_concentration_is_bivariate(self) -> None:
        with pytest.raises(ValueError, match="bivariate"):
            tail_concentration(copula=rc.ClaytonCopula(2.0, dim=3))
        with pytest.raises(ValueError, match="bivariate"):
            tail_concentration(rc.ClaytonCopula(2.0, dim=3).rvs(50, random_state=0))

    def test_pickands_plot_refuses_a_non_extreme_value_copula(self) -> None:
        with pytest.raises(ValueError, match="not an extreme-value copula"):
            pickands_plot(rc.ClaytonCopula(2.0))


class TestStructurePlots:
    """The three added for the constructions whose point is that dependence
    is not a single number."""

    def test_vine_trees_draws_one_panel_per_tree(self) -> None:
        vine = rc.VineCopula(
            [[rc.ClaytonCopula(3.0), rc.GumbelCopula(2.5)], [rc.FrankCopula(4.0)]],
            structure="D",
        )
        grid = vine_trees(vine)
        assert len(grid) == 2
        assert grid[0].get_title() == "tree 1"
        assert grid[1].get_title() == "tree 2"

    def test_vine_trees_labels_every_edge_with_its_family(self) -> None:
        vine = rc.VineCopula(
            [[rc.ClaytonCopula(3.0), rc.GumbelCopula(2.5)], [rc.FrankCopula(4.0)]],
            structure="D",
        )
        grid = vine_trees(vine)
        first = " ".join(t.get_text() for t in grid[0].texts)
        assert "Clayton" in first and "Gumbel" in first
        assert "Frank" in " ".join(t.get_text() for t in grid[1].texts)

    def test_vine_trees_can_be_truncated(self) -> None:
        vine = rc.VineCopula([[rc.GaussianCopula(0.5)] * (4 - k) for k in range(4)], structure="C")
        assert len(vine_trees(vine, max_trees=2)) == 2

    def test_nested_tree_shows_every_node_and_leaf(self) -> None:
        cop = NestedArchimedean(
            rc.GumbelCopula(1.5),
            children=[
                NestedArchimedean(rc.GumbelCopula(4.0), [0, 1, 2]),
                NestedArchimedean(rc.GumbelCopula(3.0), [3, 4]),
            ],
        )
        ax = nested_tree(cop)
        text = " ".join(t.get_text() for t in ax.texts)
        for leaf in range(5):
            assert str(leaf) in text
        # each node is annotated with its parameter and tau
        assert "tau" in text
        assert "4.00" in text and "3.00" in text and "1.50" in text

    def test_nested_tree_handles_a_deep_chain(self) -> None:
        cop = NestedArchimedean(
            rc.ClaytonCopula(0.4),
            [4],
            [
                NestedArchimedean(
                    rc.ClaytonCopula(1.5),
                    [3],
                    [NestedArchimedean(rc.ClaytonCopula(4.0), [0, 1, 2])],
                )
            ],
        )
        ax = nested_tree(cop)
        assert len(ax.collections) >= 8  # five leaves plus three nodes

    def test_dependence_heatmap_annotates_the_numbers(self) -> None:
        cop = NestedArchimedean(
            rc.GumbelCopula(1.5),
            children=[
                NestedArchimedean(rc.GumbelCopula(4.0), [0, 1, 2]),
                NestedArchimedean(rc.GumbelCopula(3.0), [3, 4]),
            ],
        )
        tau = cop.tau_matrix()
        ax = dependence_heatmap(tau, names=list("abcde"))
        text = {t.get_text() for t in ax.texts}
        assert f"{tau[0, 1]:.2f}" in text  # within the first block
        assert f"{tau[0, 3]:.2f}" in text  # across blocks
        assert [label.get_text() for label in ax.get_xticklabels()] == list("abcde")

    def test_dependence_heatmap_rejects_a_non_square_matrix(self) -> None:
        with pytest.raises(ValueError, match="square matrix"):
            dependence_heatmap(np.zeros((2, 3)))
