"""
Tests for src/features/build_features.py.

One test (test_add_market_implied_probs) is written out fully as a worked
example of the pattern: build a tiny synthetic DataFrame with known inputs,
call the function, assert on hand-calculated expected outputs. Use the same
pattern for the rest as you implement each function — testing on a handful
of rows you can verify by hand is much faster to debug than testing on the
full dataset.

Run with: pytest
"""

import pandas as pd
import pytest

from src.features.build_features import (
    add_market_implied_probs,
    compute_elo_ratings,
)


def test_add_market_implied_probs():
    # Odds chosen so the math is easy to verify by hand:
    # raw probs = 1/2, 1/4, 1/4 -> sum (overround) = 1.0 exactly, so
    # normalized probs should equal the raw probs unchanged.
    matches = pd.DataFrame({
        "odds_home": [2.0],
        "odds_draw": [4.0],
        "odds_away": [4.0],
    })

    result = add_market_implied_probs(matches)

    assert result.loc[0, "prob_home"] == pytest.approx(0.5)
    assert result.loc[0, "prob_draw"] == pytest.approx(0.25)
    assert result.loc[0, "prob_away"] == pytest.approx(0.25)
    # normalized probabilities must sum to 1
    total = result.loc[0, ["prob_home", "prob_draw", "prob_away"]].sum()
    assert total == pytest.approx(1.0)

def test_add_market_implied_probs_removes_overround():
    # raw probs: 1/2 + 2/7 + 1/4 = 29/28 (overround > 1)
    matches = pd.DataFrame({
        "odds_home": [2.0],
        "odds_draw": [3.5],
        "odds_away": [4.0],
    })

    result = add_market_implied_probs(matches)

    # normalized = raw / (29/28): 14/29, 8/29, 7/29
    assert result.loc[0, "prob_home"] == pytest.approx(14 / 29)
    assert result.loc[0, "prob_draw"] == pytest.approx(8 / 29)
    assert result.loc[0, "prob_away"] == pytest.approx(7 / 29)


def test_compute_elo_ratings_first_match_uses_initial_rating():
    """
    TODO(evan): flesh this out once compute_elo_ratings is implemented.
    A team's very first match in the dataset should get `initial_rating`
    as its pre-match Elo (there's no history yet to have moved it).
    """
    pass


def test_compute_elo_ratings_updates_after_a_win():
    """
    TODO(evan): build a 2-row synthetic dataset for the same two teams
    (Team A beats Team B in match 1), and assert Team A's Elo going into
    match 2 is HIGHER than initial_rating, and Team B's is LOWER.
    """
    pass


def test_add_rolling_form_excludes_current_match():
    """
    TODO(evan): the most important test in this file. Construct a tiny
    sequence of matches for one team and assert that the rolling-form
    features for match N do NOT include match N's own result — only
    matches strictly before it. This is the leakage check.
    """
    pass
