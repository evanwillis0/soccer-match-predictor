# Soccer Match Outcome Predictor

Predicting English Premier League match outcomes (home win / draw / away
win) from historical results, team form, and a custom Elo rating system —
benchmarked against real bookmaker odds instead of an arbitrary baseline.

**Status:** 🚧 in progress — see roadmap below.

## Why this project

Built to demonstrate an end-to-end ML workflow: real messy data, careful
feature engineering with no time-leakage, a rigorous chronological
evaluation, and an honest comparison against a genuinely hard baseline (a
market that's actively priced by professionals). Match outcome prediction
is a well-studied, well-understood problem, which makes it a good venue to
show engineering rigor rather than to claim a novel result.

## Data

Source: [football-data.co.uk](https://www.football-data.co.uk) — free
historical match results and closing Bet365 odds for the English Premier
League, no API key required. Modeled range: 2016-17 through 2025-26 (10
seasons, 3,800 matches), plus 2015-16 as a warm-up season that builds
Elo/form/head-to-head history but isn't modeled itself.

```
python -m src.data.download_data   # pulls raw season CSVs into data/raw/
python -m src.data.clean_data      # consolidates into data/processed/matches.csv
```

## Project structure

```
soccer-match-predictor/
├── data/
│   ├── raw/            # untouched season CSVs from football-data.co.uk
│   └── processed/      # matches.csv — cleaned, consolidated, chronological
├── notebooks/
│   └── 01_eda.ipynb    # Week 1: exploratory analysis
├── src/
│   ├── data/            # download + clean scripts
│   ├── features/         # feature engineering (form, rest, Elo, head-to-head, market probs)
│   └── models/            # training + evaluation
├── tests/                # pytest — mainly leakage checks on the features
├── app/                  # Week 4: FastAPI + Streamlit demo
└── requirements.txt
```

## Setup

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Roadmap

- [x] **Week 1 — Data & EDA:** download + clean the season CSVs, explore
      outcome distribution, home advantage, goals distribution, and how
      well-calibrated the betting market already is (`notebooks/01_eda.ipynb`)
- [ ] **Week 2 — Feature engineering:** rolling team form, head-to-head
      history, rest days, a from-scratch Elo rating system, and
      market-implied probabilities (`src/features/build_features.py`).
      *All features built and tested; tuning Elo's K and season
      regression remains.*
- [ ] **Week 3 — Modeling:** chronological train/val/test split; baselines
      (majority class, market odds); logistic regression → random forest →
      XGBoost, evaluated on accuracy and log loss (`src/models/`)
- [ ] **Week 4 — Deployment:** FastAPI prediction endpoint + Streamlit demo,
      deployed live (`app/`)

## Results

*(fill in once Week 3 is done)*

| Model                       | Accuracy | Log loss |
|------------------------------|----------|----------|
| Majority class baseline      | TBD      | TBD      |
| Market (bookmaker odds)      | TBD      | TBD      |
| Logistic regression          | TBD      | TBD      |
| Random forest                | TBD      | TBD      |
| XGBoost                      | TBD      | TBD      |

## Features

Every feature uses only information available before kickoff
(`build_feature_table()` in `src/features/build_features.py`).

| Feature | Columns | What it is |
|---|---|---|
| Rolling form | `home/away_form_pts`, `_goals_for`, `_goals_against` | Per-game averages over each team's last 5 matches (any venue) |
| Rest days | `home/away_rest_days` | Days since each team's previous match, capped at 14 |
| Elo rating | `home_elo`, `away_elo` | Pre-match ratings from a from-scratch Elo system (K=20, home advantage 60) |
| Head-to-head | `h2h_pts` | Home team's average points in its last 5 meetings with this opponent, shrunk toward the league average |
| Market (benchmark) | `prob_home/draw/away` | Bet365 implied probabilities with the overround removed. Kept apart from the features: it's the baseline to beat |

## Key design decisions

### Evaluation

- **Chronological splits only.** Random train/test splits leak future
  information through the rolling-form and Elo features — every model
  here is trained on earlier seasons and evaluated on strictly later ones.
- **The betting market is the real baseline**, not just the majority-class
  guess. Beating a liquid betting market is genuinely hard — the honest
  goal here is to understand *how close* a from-scratch model can get, not
  to claim profitability.

### Elo ratings

- **Draws count as half a win for both teams.**
  The Elo rates draws differently for each team depending on the current Elo of the teams that drew. If a lower-Elo team drew with a higher-Elo team, the lower Elo would go up and vice versa for the higher. Keeping the chess convention is fine because Elo's job is to measure team strength, not to predict draws. Draws being common is handled later by the classifier, which can learn that evenly rated matchups draw more often.
- **Ratings move partway back toward 1500 each summer (33%).**
  A carried-over rating ignores any summer changes to the team such as manager, injuries, signings, or other circumstances, assuming the team is in the exact same form it was in the past season. Most of the time this isn't true, so moving the Elo toward average at the start, to be safe and account for these changes, lets the Elo adjust as needed based on results from the current season as teams' in-season form is determined. 
- **Promoted teams inherit the relegated teams' average rating.**
  Promoted teams inherit the relegated average rating, as promoted teams averaged 0.91 pts/game compared to the league average of 1.38, so it's safe to assume that the promoted teams' Elo would be close to that of the relegated team, as 13 of 24 promoted teams in the past 10 years went straight back down. This inheritance also keeps the league average Elo around 1500, as the teams' Elo ratings are replaced, so the average holds.

### Head-to-head

- **Meetings at any venue count, not just the same ground.**
  Counting meetings at any venue between the two teams, rather than only past home or away matchups, gives us a larger sample size for the model to make a better prediction. Home advantage is not lost: every feature is split into home and away columns, so the model knows which team is at home and learns the home-win rate directly.
- **Thin histories are shrunk toward the league average:**
  `(H2H points + 1.39 × 2) / (meetings + 2)`. One big win or loss shouldn't read as dominance for one side or the other, since a lot of soccer is luck and randomness. A single result could be a fluke or down to extenuating circumstances, which would feed the model misinformation about the actual relationship between the teams. So the formula acts as if every pair also has 2 extra "average" meetings worth 1.39 pts each (league-average points per game). With no history the value is exactly league average, with 1–2 meetings it stays close to average, and with the full 5 meetings the real results dominate.

### Feature table

- **2015-16 is a warm-up season.**
  We use an entire season as a warm-up for the features (Elo, form, head-to-head); the model itself never sees 2015-16. Without it, in the 2016-17 season all teams would start with the same average Elo and no form or H2H history, which could cause inaccurate predictions in the first few weeks as teams get into form and their Elo adjusts properly. The warm-up season means every feature already has real history before we try to predict matches.
- **Promoted teams' missing form is filled with promoted-team averages, not dropped.**
  Dropping these rows would mean the model never trains on a promoted team's first matches, yet every August it has to predict exactly those. Filling missing form with what promoted teams typically do (0.93 pts, 1.00 scored, 1.74 conceded per game) keeps those matches in training with a realistic estimate.
- **Same-match stats (goals, shots, corners) are excluded.**
  If same-match stats were included, then the model would be making a prediction for the game based on the stats of that very game that it's trying to predict, so it would be overly accurate and not a genuine prediction based on past stats. We could still use these stats by adding them to the teams' rolling form, using past matches' stats rather than the current match's to predict it (future improvement).
- **Every number used as a default (1.39 pts, promoted-team averages) is
  computed on training seasons only.** 
  If these defaults were computed from every season, a little information from the validation and test seasons would seep into the features, making the final test score slightly optimistic. Computing them on training seasons only keeps the test set genuinely untouched, even though the leak would be tiny.

### Leakage check

Training the same model on correct pre-match Elo vs a deliberately leaked
post-match Elo (validated on 2024-25):

| Elo feature | 2024-25 accuracy | 2024-25 log loss |
|---|---|---|
| Pre-match (correct) | 55.8% | 0.979 |
| Post-match (leaked) | 58.7% | 0.911 |

The leaked number considered the Elo of the teams after the match to predict the match itself between the two, so it was inherently more accurate because it already knew the Elo change after the match. The small size of the gain makes the leaks seem less important, which can make them hard to catch as an error since it's not a big difference, making them dangerous. 

## Hypotheses to test (Week 3)

- **Head-to-head matters most between evenly matched teams.** 
  If two teams are of similar Elo, then head-to-head matters most in predicting the outcome, as it relatively compares the teams' match histories rather than comparing them to the league as a whole using the Elo system. The plan is to train each model twice, with and without `h2h_pts`, on the training seasons, then compare log loss on the validation season, broken down by Elo gap (close vs lopsided matches). If the hypothesis holds, the gain should concentrate in close matches and be larger for random forest and XGBoost than for logistic regression, because tree models can learn "H2H matters when Elo is close" on their own.

## Live demo

*(link once deployed — Week 4)*
