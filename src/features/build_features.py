"""
Feature engineering for the match outcome model.

>>> THIS IS YOUR FILE TO WRITE (Week 2). <<<
The functions below are stubbed out with signatures, docstrings, and the
key gotcha called out for each one. Implement them one at a time and add a
matching test in tests/test_features.py as you go.

THE GOLDEN RULE FOR ALL OF THESE: every feature for a given match must only
use information that was actually available *before that match kicked off*.
`matches` is assumed sorted chronologically (build_matches_table() already
sorts by date) — lean on that ordering, and never use groupby().mean() over
a team's *entire* history including future matches, or you'll leak the
answer into your own training data.
"""

import pandas as pd


def add_rolling_form(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Add columns for each team's form over their last `window` matches, computed
    *before* the current match: points earned (3/1/0), goals scored, goals
    conceded. You'll want one set for the home team and one for the away team,
    e.g. columns like `home_form_pts`, `home_form_goals_for`,
    `home_form_goals_against`, and the mirrored `away_*` versions.

    Hint: because a team appears in the `home_team` column in some rows and
    `away_team` in others, it's easiest to first build a long "team-perspective"
    table (one row per team per match, regardless of home/away), compute the
    rolling stats with `.shift(1)` before `.rolling(window)` (the shift is what
    keeps the current match's own result out of its own rolling average), then
    join those numbers back onto the original home/away match table.

    Args:
        matches: chronologically-sorted match table (see clean_data.py)
        window: number of prior matches to average over

    Returns:
        matches with new form columns added
    """
    raise NotImplementedError


def add_head_to_head(matches: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    Add a feature summarizing the home team's results in their last
    `lookback` meetings against this specific opponent (e.g. average points
    earned in that head-to-head history). Same leakage rule applies: only
    matches strictly before the current one.
    """
    raise NotImplementedError


def add_rest_days(matches: pd.DataFrame, max_rest_days = 14) -> pd.DataFrame:
    """
    Add `home_rest_days` / `away_rest_days`: number of days since each team's
    previous match. Fatigue/rotation is a real factor in outcome prediction
    and this is a cheap, honest feature to compute.
    """
    matches = matches.copy()
    home = pd.DataFrame({"match_id": matches.index, "date": matches["date"],
                         "team": matches["home_team"], "side": "home"})
    away = pd.DataFrame({"match_id": matches.index, "date": matches["date"],
                         "team": matches["away_team"], "side": "away"})
    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team", "date"])
    long["rest_days"] = long.groupby("team")["date"].diff().dt.days
    for side in ["home", "away"]:
        part = long[long["side"] == side].set_index("match_id")["rest_days"]
        matches[f"{side}_rest_days"] = part.clip(upper=max_rest_days).fillna(max_rest_days)
    return matches


def compute_elo_ratings(
    matches: pd.DataFrame,
    k: float = 20.0,
    initial_rating: float = 1500.0,
    home_advantage: float = 60.0,
) -> pd.DataFrame:
    """
    Compute a running Elo rating for every team and attach each match's
    *pre-match* home/away ratings as features (`home_elo`, `away_elo`).

    This is the classic Elo update, adapted for soccer's three outcomes:
      1. Expected score for the home team:
           E_home = 1 / (1 + 10 ** ((away_rating - (home_rating + home_advantage)) / 400))
      2. Actual score: 1.0 for a home win, 0.5 for a draw, 0.0 for a home loss
         (this is the standard chess-Elo convention of treating a draw as
         "half a win" for both sides — think about whether that's the right
         call for soccer, where draws are much more common than in chess, or
         whether you want to weight it differently. Worth a sentence in your
         README either way.)
      3. Update: new_home_rating = home_rating + k * (actual_score - E_home)
         and symmetrically for the away team using (1 - actual_score).

    Process matches in chronological order, maintaining a running
    `{team_name: rating}` dict, initialized to `initial_rating` for every
    team the first time you see them. For each match: record the CURRENT
    (pre-update) ratings as that match's `home_elo`/`away_elo` features,
    THEN apply the update for the next match.

    Args:
        matches: chronologically-sorted match table
        k: how much a single result moves a team's rating (higher = more volatile)
        initial_rating: starting rating for a team with no history
        home_advantage: rating points added to the home team's effective rating

    Returns:
        matches with `home_elo` and `away_elo` columns added
    """
    raise NotImplementedError


def add_market_implied_probs(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the Bet365 decimal odds (odds_home/odds_draw/odds_away) into
    implied probabilities, and normalize them to remove the bookmaker's
    overround (the "vig" — decimal odds imply probabilities that sum to
    slightly more than 1, which is the bookmaker's built-in margin).

    Steps:
      1. raw_prob_x = 1 / odds_x for each outcome
      2. overround = sum of the three raw probs (will be > 1, e.g. ~1.05)
      3. normalized_prob_x = raw_prob_x / overround

    These normalized probabilities are your BENCHMARK — the number your
    model's predictions need to beat (or come close to) in Week 3. It's
    fine, and expected, to also feed them in as model features — just be
    explicit in your write-up about the difference between "using the
    market as an input feature" vs "using the market as the baseline to
    beat", since those are different experiments.
    """
    matches = matches.copy()
    raw_home = 1 / matches["odds_home"]
    raw_draw = 1 / matches["odds_draw"]
    raw_away = 1 / matches["odds_away"]
    overround = raw_home + raw_draw + raw_away
    matches["prob_home"] = raw_home / overround
    matches["prob_draw"] = raw_draw / overround
    matches["prob_away"] = raw_away / overround
    return matches


def build_feature_table(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the pipeline: run all the add_* functions above in order
    and return one feature table ready for modeling. Drop rows from each
    team's first `window` matches of the dataset if their rolling-form
    features end up NaN (no history yet) — decide on and document that
    cutoff, don't just silently dropna() everything.
    """
    raise NotImplementedError
