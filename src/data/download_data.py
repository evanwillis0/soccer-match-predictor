"""
Download historical English Premier League match data (results + betting odds)
from football-data.co.uk.

Data source: https://www.football-data.co.uk/englandm.php
Free, no API key required. Each file covers one season of one league.

NOTE: football-data.co.uk is blocked from Anthropic's cloud sandbox egress
proxy, so this script needs to be run from your own machine (or anywhere
with normal internet access), not inside the Claude cloud container.

Usage:
    python -m src.data.download_data
    python -m src.data.download_data --seasons 2223 2324 2425 2526
"""

import argparse
from pathlib import Path

import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

# League codes on football-data.co.uk. E0 = English Premier League.
# (Feel free to add more, e.g. E1 = Championship, SP1 = La Liga, D1 = Bundesliga)
LEAGUE_CODE = "E0"

# Season codes are two two-digit years concatenated, e.g. "2425" = 2024-25.
# Default: the 10 most recently *completed* EPL seasons as of Sep 2026.
DEFAULT_SEASONS = [
    "1617", "1718", "1819", "1920", "2021",
    "2122", "2223", "2324", "2425", "2526",
]

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def download_season(season: str, league: str = LEAGUE_CODE, out_dir: Path = RAW_DIR) -> Path:
    """Download one season's CSV if we don't already have it. Returns the local path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{season}_{league}.csv"
    if dest.exists():
        print(f"  [skip] {dest.name} already downloaded")
        return dest

    url = BASE_URL.format(season=season, league=league)
    print(f"  [get]  {url}")
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()

    # football-data.co.uk CSVs are latin-1 encoded, not utf-8 — decode explicitly
    # so we can re-save clean utf-8 and avoid mojibake later on.
    text = resp.content.decode("latin-1")
    dest.write_text(text, encoding="utf-8")
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons", nargs="+", default=DEFAULT_SEASONS,
        help="Season codes to download, e.g. 2324 2425 2526",
    )
    parser.add_argument("--league", default=LEAGUE_CODE, help="League code (default: E0 = EPL)")
    args = parser.parse_args()

    print(f"Downloading {len(args.seasons)} season(s) of league '{args.league}' to {RAW_DIR}")
    paths = [download_season(s, args.league) for s in args.seasons]
    print(f"\nDone. {len(paths)} files in {RAW_DIR}")


if __name__ == "__main__":
    main()
