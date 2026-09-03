"""
Model training for the match outcome predictor.

>>> THIS IS YOUR FILE TO WRITE (Week 3). <<<

Target: `result` (H/D/A) — a 3-class classification problem.

REQUIRED: use a chronological split, never a random one.
    e.g. train on seasons 2016-17 through 2023-24, validate on 2024-25,
    hold out 2025-26 as the final test set you only touch once. A random
    shuffled split would let the model "see the future" (a team's Elo/form
    in April can leak information about matches in the same season's
    earlier months via the rolling stats), which would make your reported
    accuracy meaningless.

Suggested structure:
    1. load_feature_table() -> uses build_features.build_feature_table()
    2. chronological_split(df) -> (train, val, test) by season
    3. baseline models to compare against:
         - majority class (always predict the most common outcome, usually
           home win — this is your "dumbest possible model" floor)
         - market-implied probabilities from odds (your "hardest to beat"
           ceiling — see add_market_implied_probs in build_features.py)
    4. real models, roughly in increasing complexity:
         - multinomial LogisticRegression (scikit-learn) as your first real
           model — interpretable coefficients are good for your README
         - RandomForestClassifier
         - XGBoost's XGBClassifier (objective="multi:softprob")
    5. save the best model with joblib so app/ can load it later:
         joblib.dump(model, "models/best_model.pkl")

Remember to scale/encode consistently between train and val/test (fit any
scaler/encoder on train only, then .transform() — never .fit() — on val/test).
"""

# TODO(evan): implement. Start by getting ONE model (logistic regression)
# trained end-to-end and scored against the majority-class baseline before
# you touch anything fancier — get the harness right first.
