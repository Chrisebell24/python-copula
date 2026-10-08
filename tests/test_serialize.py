"""Regression tests for :mod:`rcopula.serialize`: fit results and clear errors."""

from __future__ import annotations

import json

import numpy as np
import pytest

import rcopula as rc
from rcopula.serialize import from_dict, from_json, to_dict, to_json


def test_fit_result_serialises_its_fitted_copula() -> None:
    u = rc.GumbelCopula(2.0).rvs(300, random_state=0)
    result = rc.fit(rc.GumbelCopula(), u)
    text = to_json(result)
    reloaded = from_json(text)
    assert reloaded == result.copula
    np.testing.assert_array_equal(reloaded.logpdf(u), result.copula.logpdf(u))
    fit = json.loads(text)["fit"]
    assert fit["result"] == "CopulaFitResult"
    assert fit["method"] == result.method
    assert fit["n_obs"] == 300
    assert fit["params"] == [float(p) for p in result.params]
    assert fit["loglik"] == result.loglik
    assert np.asarray(fit["cov_params"]).shape == (1, 1)


def test_discrete_fit_result_serialises() -> None:
    from rcopula.discrete import fit_discrete

    rng = np.random.default_rng(0)
    x = rng.poisson(2.0, size=(80, 2))
    from scipy import stats

    margins = [stats.poisson(2.0), stats.poisson(2.0)]
    result = fit_discrete(x, rc.FrankCopula(1.0), margins)
    document = to_dict(result)
    assert document["fit"]["result"] == "DiscreteFitResult"
    assert from_dict(document) == result.copula
    json.dumps(document, allow_nan=False)


def test_non_copula_raises_type_error() -> None:
    with pytest.raises(TypeError, match="Copula or a fit result"):
        to_json(object())
    with pytest.raises(TypeError, match="Copula or a fit result"):
        to_dict({"copula": 1})
