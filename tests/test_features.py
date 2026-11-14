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
    add_rest_days,
    compute_elo_ratings,
    add_rolling_form,

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

def test_add_rest_days():
    matches = pd.DataFrame({
        "date": pd.to_datetime(["2024-08-10", "2024-08-14", "2024-08-20"]),
        "home_team": ["A", "B", "A"],
        "away_team": ["B", "C", "C"],
    })
    result = add_rest_days(matches)
    assert result.loc[0, "home_rest_days"] == 14       # A's first match: no history, treated as rested
    assert result.loc[1, "home_rest_days"] == 4        # B: Aug 10 (away) -> Aug 14
    assert result.loc[2, "home_rest_days"] == 10       # A: Aug 10 -> Aug 20
    assert result.loc[2, "away_rest_days"] == 6        # C: Aug 14 -> Aug 20


def test_add_rest_days_caps_long_gaps():
    matches = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-03-01"]),
        "home_team": ["A", "A"],
        "away_team": ["B", "C"],
    })
    result = add_rest_days(matches)
    assert result.loc[1, "home_rest_days"] == 14       # a 60-day gap is capped

# Hand-calculated Elo change when two 1500 teams meet and the HOME team wins
# (k=20, home_advantage=60):
#   E_home = 1 / (1 + 10 ** ((1500 - 1560) / 400)) = 0.58550
#   change = 20 * (1 - 0.58550) = 8.2900
HOME_WIN_CHANGE = 20 * (1 - 1 / (1 + 10 ** (-60 / 400)))


def test_compute_elo_ratings_first_match_uses_initial_rating():
    matches = pd.DataFrame({
        "home_team": ["A"], "away_team": ["B"], "result": ["H"],
    })
    result = compute_elo_ratings(matches)
    # pre-match ratings: the match's own result must not have moved them
    assert result.loc[0, "home_elo"] == pytest.approx(1500.0)
    assert result.loc[0, "away_elo"] == pytest.approx(1500.0)


def test_compute_elo_ratings_updates_after_a_win():
    matches = pd.DataFrame({
        "home_team": ["A", "A"], "away_team": ["B", "B"], "result": ["H", "D"],
    })
    result = compute_elo_ratings(matches)
    assert result.loc[1, "home_elo"] > 1500
    assert result.loc[1, "away_elo"] < 1500
    assert result.loc[1, "home_elo"] == pytest.approx(1500 + HOME_WIN_CHANGE)
    # zero-sum: A gains exactly what B loses
    assert result.loc[1, "away_elo"] == pytest.approx(1500 - HOME_WIN_CHANGE)


def test_compute_elo_ratings_regresses_between_seasons():
    matches = pd.DataFrame({
        "season": ["s1", "s2"],
        "home_team": ["A", "A"], "away_team": ["B", "B"], "result": ["H", "H"],
    })
    result = compute_elo_ratings(matches, season_regression=0.5)
    # A ended s1 at 1500 + change; half of that gain is wiped over the summer
    assert result.loc[1, "home_elo"] == pytest.approx(1500 + 0.5 * HOME_WIN_CHANGE)
    assert result.loc[1, "away_elo"] == pytest.approx(1500 - 0.5 * HOME_WIN_CHANGE)


def test_compute_elo_ratings_promoted_team_inherits_relegated_rating():
    # B plays only in s1 (relegated); C appears in s2 (promoted)
    matches = pd.DataFrame({
        "season": ["s1", "s2"],
        "home_team": ["A", "A"], "away_team": ["B", "C"], "result": ["H", "H"],
    })
    result = compute_elo_ratings(matches, season_regression=0.5)
    # C starts at B's regressed end-of-season rating, not at 1500
    assert result.loc[1, "away_elo"] == pytest.approx(1500 - 0.5 * HOME_WIN_CHANGE)
    # ...which keeps the league average at exactly 1500
    assert (result.loc[1, "home_elo"] + result.loc[1, "away_elo"]) / 2 == pytest.approx(1500)


def test_add_rolling_form_excludes_current_match():
    # Team A plays four home matches: W 2-0, D 1-1, L 0-3, W 1-0
    matches = pd.DataFrame({
        "date": pd.to_datetime(["2024-08-01", "2024-08-08", "2024-08-15", "2024-08-22"]),
        "home_team": ["A", "A", "A", "A"],
        "away_team": ["B", "C", "D", "E"],
        "home_goals": [2, 1, 0, 1],
        "away_goals": [0, 1, 3, 0],
    })
    result = add_rolling_form(matches, window=2)

    # not enough history yet -> NaN
    assert pd.isna(result.loc[0, "home_form_pts"])
    assert pd.isna(result.loc[1, "home_form_pts"])
    # match 2 uses matches 0 and 1 only: points (3+1)/2, goals for (2+1)/2, against (0+1)/2
    assert result.loc[2, "home_form_pts"] == pytest.approx(2.0)
    assert result.loc[2, "home_form_goals_for"] == pytest.approx(1.5)
    assert result.loc[2, "home_form_goals_against"] == pytest.approx(0.5)
    # match 3 uses matches 1 and 2: points (1+0)/2
    assert result.loc[3, "home_form_pts"] == pytest.approx(0.5)
def test_add_rolling_form_resets_after_long_gap():
    matches = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15",
                                "2024-12-01", "2024-12-08"]),
        "home_team": ["A"] * 5,
        "away_team": ["B", "C", "D", "E", "F"],
        "home_goals": [1, 1, 1, 1, 1],
        "away_goals": [0, 0, 0, 0, 0],
    })
    result = add_rolling_form(matches, window=2)
    assert result.loc[2, "home_form_pts"] == pytest.approx(3.0)   # normal: two wins before
    assert pd.isna(result.loc[3, "home_form_pts"])                # 321-day gap -> history reset
    assert pd.isna(result.loc[4, "home_form_pts"])                # only one match in the new stint