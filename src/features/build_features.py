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

from collections import defaultdict, deque

import pandas as pd


def add_rolling_form(matches: pd.DataFrame, window: int = 5, max_gap_days: int = 120) -> pd.DataFrame:
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
    matches = matches.copy()
    home = pd.DataFrame({"match_id": matches.index, "date": matches["date"],
                         "team": matches["home_team"], "side": "home",
                         "goals_for": matches["home_goals"],
                         "goals_against": matches["away_goals"]})
    away = pd.DataFrame({"match_id": matches.index, "date": matches["date"],
                         "team": matches["away_team"], "side": "away",
                         "goals_for": matches["away_goals"],
                         "goals_against": matches["home_goals"]})
    long = pd.concat([home, away], ignore_index=True)
    long["points"] = (3 * (long["goals_for"] > long["goals_against"])
                      + 1 * (long["goals_for"] == long["goals_against"]))
    long = long.sort_values(["team", "date"])
    gap = long.groupby("team")["date"].diff().dt.days
    long["stint"] = (gap > max_gap_days).groupby(long["team"]).cumsum()

    stats = ["points", "goals_for", "goals_against"]
    form = long.groupby(["team", "stint"])[stats].transform(
        lambda s: s.shift(1).rolling(window).mean()
    )
    form = form.rename(columns={"points": "form_pts",
                                "goals_for": "form_goals_for",
                                "goals_against": "form_goals_against"})
    long = long.join(form)

    for side in ["home", "away"]:
        part = long[long["side"] == side].set_index("match_id")[list(form.columns)]
        part.columns = [f"{side}_{c}" for c in part.columns]
        matches = matches.join(part)
    return matches


def add_head_to_head(
    matches: pd.DataFrame,
    lookback: int = 5,
    prior_pts: float = 1.39,
    prior_weight: float = 2.0,
) -> pd.DataFrame:
    """
    Add `h2h_pts`: the home team's average points (3/1/0) in its last
    `lookback` meetings with this opponent, at any venue, using only
    matches strictly before the current one.

    Thin histories (promoted teams, or the dataset starting in 2016) are
    handled by shrinking toward the league average, as if every pair had
    `prior_weight` extra meetings that each earned `prior_pts`:

        h2h_pts = (sum of H2H points + prior_pts * prior_weight) / (meetings + prior_weight)

    So 0 meetings -> exactly `prior_pts`, a single win doesn't look like
    dominance, and 5 meetings are mostly trusted. The value is always
    defined, so there are no NaN rows to drop.

    Design decisions:
      - Any venue, not same-venue only: roughly double the sample, and home
        advantage is already captured by Elo.
      - prior_pts = 1.39 is league-average points per team per game on the
        TRAINING seasons only (2016-17 to 2023-24), equal to
        (3 - draw rate) / 2.
    """
    matches = matches.copy()
    history: dict[frozenset, deque] = defaultdict(lambda: deque(maxlen=lookback))
    h2h_pts = []

    for home, away, home_goals, away_goals in zip(
        matches["home_team"], matches["away_team"], matches["home_goals"], matches["away_goals"]
    ):
        past = history[frozenset((home, away))]

        # Feature from PAST meetings only...
        earned = sum(meeting[home] for meeting in past)
        h2h_pts.append((earned + prior_pts * prior_weight) / (len(past) + prior_weight))

        # ...THEN record this meeting for future ones.
        if home_goals > away_goals:
            past.append({home: 3, away: 0})
        elif home_goals < away_goals:
            past.append({home: 0, away: 3})
        else:
            past.append({home: 1, away: 1})

    matches["h2h_pts"] = h2h_pts
    return matches


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
    season_regression: float = 0.33,
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

    Design decisions (see README "Key design decisions"):
      - Draws count as 0.5 for both sides. Elo's job is measuring strength;
        predicting draws is left to the classifier.
      - Between seasons, every rating is pulled `season_regression` of the
        way back toward `initial_rating` (summer transfers, manager changes).
      - Promoted teams (any team not in the previous season, including
        teams returning after relegation) start at the average rating of
        the teams that left the league, after regression. The new season's
        team list is known before it kicks off, so this isn't leakage.
        Because Elo updates are zero-sum and newcomers replace leavers one
        for one, the league-average rating stays at `initial_rating`.
      - Teams in the first season of the dataset all start at `initial_rating`.

    Args:
        matches: chronologically-sorted match table. Uses the `season`
            column to detect season boundaries; without one, all matches
            are treated as a single season.
        k: how much a single result moves a team's rating (higher = more volatile)
        initial_rating: starting rating for a team with no history
        home_advantage: rating points added to the home team's effective rating
        season_regression: fraction (0-1) of the way each rating moves back
            toward `initial_rating` between seasons. 0 = carry over fully.

    Returns:
        matches with `home_elo` and `away_elo` columns added
    """
    matches = matches.copy()
    seasons = matches["season"] if "season" in matches.columns else pd.Series(0, index=matches.index)
    actual_home = {"H": 1.0, "D": 0.5, "A": 0.0}

    ratings: dict[str, float] = {}
    home_elo, away_elo = [], []
    current_season = None

    for season, home, away, result in zip(
        seasons, matches["home_team"], matches["away_team"], matches["result"]
    ):
        if season != current_season:
            if current_season is not None:
                # Season boundary: regress everyone toward the mean, then
                # swap leavers for newcomers.
                in_league = seasons == season
                new_teams = set(matches.loc[in_league, "home_team"]) | set(matches.loc[in_league, "away_team"])
                ratings = {t: initial_rating + (1 - season_regression) * (r - initial_rating)
                           for t, r in ratings.items()}
                leavers = [ratings[t] for t in ratings if t not in new_teams]
                newcomer_rating = sum(leavers) / len(leavers) if leavers else initial_rating
                ratings = {t: ratings.get(t, newcomer_rating) for t in new_teams}
            current_season = season

        home_rating = ratings.setdefault(home, initial_rating)
        away_rating = ratings.setdefault(away, initial_rating)

        # Record PRE-match ratings as this match's features...
        home_elo.append(home_rating)
        away_elo.append(away_rating)

        # ...THEN update for future matches.
        expected_home = 1 / (1 + 10 ** ((away_rating - (home_rating + home_advantage)) / 400))
        change = k * (actual_home[result] - expected_home)
        ratings[home] = home_rating + change
        ratings[away] = away_rating - change

    matches["home_elo"] = home_elo
    matches["away_elo"] = away_elo
    return matches


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
