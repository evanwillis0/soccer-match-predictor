"""
Evaluation utilities for comparing your models against each other and
against the betting market.

>>> THIS IS YOUR FILE TO WRITE (Week 3). <<<

Metrics to report for every model (baselines included) on the same test set:
    - accuracy — easy to explain, but weak: rewards overconfident wrong
      guesses and doesn't care how close your probabilities were
    - log loss (sklearn.metrics.log_loss) — the metric that actually
      matters here, since it scores your full probability distribution
      against what happened, not just your top pick. This is also the
      metric that lets you compare directly against the market's implied
      probabilities on equal footing.
    - a confusion matrix (3x3: H/D/A) — useful for spotting whether your
      model just refuses to ever predict draws, which is a very common
      failure mode for this exact problem (draws are the minority class
      and hardest to call)

Suggested output: a small results table (one row per model, columns =
accuracy / log_loss), plus a short written paragraph in your README stating
plainly whether you beat the market baseline or not, and by how much. If you
don't beat it, that's a legitimate, honest result — frame it as "closed X%
of the log-loss gap between the majority-class baseline and the market" and
say why you think the market still wins (bookmakers price in injury news,
lineup news, and far more signals than a public historical dataset has).
"""

# TODO(evan): implement `evaluate_model(model, X_test, y_test)` returning a
# dict of {"accuracy": ..., "log_loss": ...}, and a `results_table(models_dict,
# X_test, y_test)` that runs it over all your models/baselines and returns a
# tidy DataFrame you can drop straight into the README.
