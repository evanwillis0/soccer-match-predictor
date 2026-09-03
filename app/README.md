# Week 4: deployment

This folder is where the FastAPI endpoint + Streamlit demo will live once
you have a trained model saved at `models/best_model.pkl` (Week 3 done).

Planned layout:
- `api.py` — FastAPI app with a `/predict` endpoint: takes `home_team`,
  `away_team`, `date` and returns win/draw/loss probabilities from the
  saved model (you'll need to recompute that match's features — form,
  Elo, etc. — as of `date`, using the same functions from
  `src/features/build_features.py`, not retrain anything).
- `streamlit_app.py` — a small UI on top of the API (or calling the
  prediction function directly): two team dropdowns, a predicted
  probability bar chart, and — nice touch — the actual market odds for
  that fixture shown side by side if you have them, so a visitor can see
  your model vs. the bookmaker at a glance.

Left empty intentionally until Week 3 is done — come back to this then.
