"""
Consolidate the raw per-season football-data.co.uk CSVs (data/raw/*.csv) into
a single clean, chronologically-sorted match table (data/processed/matches.csv).

This only does column selection / renaming / dtype cleanup — it does NOT
build any model features (rolling form, Elo, etc.). That's feature
engineering and lives in src/features/build_features.py instead, so the
two concerns stay separate and testable on their own.

Usage:
    python -m src.data.clean_data
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Raw column -> clean column name. Only columns present in every season file
# we care about; older seasons have fewer stat columns than recent ones, so
# keep this list conservative. Extend it later if you want more (e.g. HY/AY
# for cards) once you've checked which seasons actually have them.
COLUMN_MAP = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",          # 'H' / 'D' / 'A'
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_target",
    "AST": "away_shots_target",
    "HC": "home_corners",
    "AC": "away_corners",
    # Bet365 closing odds — the most consistently-present bookmaker across
    # seasons on football-data.co.uk. This is our "market benchmark".
    "B365H": "odds_home",
    "B365D": "odds_draw",
    "B365A": "odds_away",
}

REQUIRED_COLUMNS = ["date", "home_team", "away_team", "home_goals", "away_goals", "result"]


def _parse_date(series: pd.Series) -> pd.Series:
    """football-data.co.uk mixes dd/mm/yy and dd/mm/yyyy formats across seasons."""
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def load_raw_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    available = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)
    df["date"] = _parse_date(df["date"])

    # season label like "2024-25", inferred from the filename (e.g. "2425_E0.csv")
    season_code = path.stem.split("_")[0]
    df["season"] = f"20{season_code[:2]}-{season_code[2:]}"

    return df


def build_matches_table() -> pd.DataFrame:
    raw_files = sorted(RAW_DIR.glob("*.csv"))
    if not raw_files:
        raise FileNotFoundError(
            f"No raw CSVs found in {RAW_DIR}. Run `python -m src.data.download_data` first."
        )

    frames = [load_raw_file(p) for p in raw_files]
    matches = pd.concat(frames, ignore_index=True)

    missing_required = [c for c in REQUIRED_COLUMNS if c not in matches.columns]
    if missing_required:
        raise ValueError(f"Missing required columns after load: {missing_required}")

    before = len(matches)
    matches = matches.dropna(subset=REQUIRED_COLUMNS)
    dropped = before - len(matches)
    if dropped:
        print(f"Dropped {dropped} rows with missing core fields (unparseable dates, etc.)")

    matches["home_goals"] = matches["home_goals"].astype(int)
    matches["away_goals"] = matches["away_goals"].astype(int)

    matches = matches.sort_values("date").reset_index(drop=True)

    # TODO(evan): if you spot inconsistent team naming across seasons when you
    # do EDA (e.g. a club renamed/promoted under a slightly different string),
    # add a normalization map here. football-data.co.uk is usually consistent
    # for EPL, but worth checking when a team's row count looks off.

    return matches


def main():
    matches = build_matches_table()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "matches.csv"
    matches.to_csv(out_path, index=False)
    print(f"Wrote {len(matches)} matches ({matches['season'].nunique()} seasons) to {out_path}")
    print(f"Date range: {matches['date'].min().date()} to {matches['date'].max().date()}")


if __name__ == "__main__":
    main()
