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
League, no API key required. Default range: 2016-17 through 2025-26 (10
seasons, ~3,800 matches).

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
│   ├── features/         # feature engineering (rolling form, Elo, market probs)
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

- [ ] **Week 1 — Data & EDA:** download + clean the season CSVs, explore
      outcome distribution, home advantage, goals distribution, and how
      well-calibrated the betting market already is (`notebooks/01_eda.ipynb`)
- [ ] **Week 2 — Feature engineering:** rolling team form, head-to-head
      history, rest days, a from-scratch Elo rating system, and
      market-implied probabilities (`src/features/build_features.py`)
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

## Key design decisions

- **Chronological splits only.** Random train/test splits leak future
  information through the rolling-form and Elo features — every model
  here is trained on earlier seasons and evaluated on strictly later ones.
- **The betting market is the real baseline**, not just the majority-class
  guess. Beating a liquid betting market is genuinely hard — the honest
  goal here is to understand *how close* a from-scratch model can get, not
  to claim profitability.

## Live demo

*(link once deployed — Week 4)*
