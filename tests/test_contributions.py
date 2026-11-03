"""Tests for SHAP contributions and the readable top reasons."""

import numpy as np
import pandas as pd
import pytest

from f1predictor.explain.contributions import explain_race, top_reasons
from f1predictor.features.race_features import POST_QUALI_FEATURES
from f1predictor.models.position import train_model


@pytest.fixture
def trained():
    """A model where finish = grid exactly, so 'grid' must carry the explanation."""
    rng = np.random.default_rng(0)
    rows = []
    for race in range(30):
        for grid in range(1, 11):
            row = {f: float(rng.random()) for f in POST_QUALI_FEATURES}   # noise features
            row.update(session_id=race, season=2024, driver_id=f"d{grid}",
                       grid=grid, finish_position=float(grid))
            rows.append(row)
    df = pd.DataFrame(rows)
    return train_model(df), df[df.session_id == 0]


def test_contributions_add_up_to_the_prediction(trained):
    model, race = trained
    explanation = explain_race(model, race)
    per_driver = explanation.groupby("driver_id").agg(total=("contribution", "sum"),
                                                      baseline=("baseline", "first"),
                                                      predicted=("predicted", "first"))
    model_says = pd.Series(model.predict(race[POST_QUALI_FEATURES]), index=race["driver_id"])
    # baseline + sum of contributions = prediction (the defining SHAP property).
    assert np.allclose(per_driver["baseline"] + per_driver["total"], per_driver["predicted"], atol=1e-4)
    assert np.allclose(per_driver["predicted"], model_says.loc[per_driver.index], atol=1e-4)


def test_one_row_per_driver_per_feature(trained):
    model, race = trained
    assert len(explain_race(model, race)) == len(race) * len(POST_QUALI_FEATURES)


def test_top_reason_is_the_feature_that_matters(trained):
    model, race = trained
    explanation = explain_race(model, race)
    # Pole-sitter: grid explains everything and moves them FORWARD.
    assert top_reasons(explanation, "d1", n=1)[0].startswith("starting position = 1:")
    assert "forward" in top_reasons(explanation, "d1", n=1)[0]
    # Last on the grid: moved BACK.
    assert "back" in top_reasons(explanation, "d10", n=1)[0]


def test_top_reasons_sorted_by_size_and_handles_unknown():
    explanation = pd.DataFrame({
        "driver_id": ["a"] * 3,
        "feature": ["x", "y", "z"],
        "label": ["small", "big", "unknown one"],
        "value": [1.0, 2.0, float("nan")],
        "contribution": [0.1, -2.0, 0.5],
    })
    assert top_reasons(explanation, "a") == [
        "big = 2: 2.0 places forward",
        "unknown one = unknown: 0.5 places back",
        "small = 1: 0.1 places back",
    ]
