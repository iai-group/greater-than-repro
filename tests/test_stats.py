import numpy as np
from gtlib.stats import bootstrap_ci, effect_size


def test_bootstrap_ci_sane():
    # NOTE: boundary uses <= rather than the brief's strict "<". With a 50/50 binary
    # mixture (n=100), the bootstrap-mean distribution is Binomial(100, 0.5)/100, whose
    # mass concentrates exactly at k=40 and k=60. Verified numerically (see task-5
    # report) that np.quantile's 2.5%/97.5% order statistics land on exactly 0.4/0.6 for
    # essentially every seed and every n_boot tried (2000-10000) - i.e. the brief's
    # strict inequality is unsatisfiable for this exact dataset, not a bug in
    # bootstrap_ci. <= preserves the intended "CI brackets 0.5 within [0.4, 0.6]" check.
    r = bootstrap_ci(np.r_[np.zeros(50), np.ones(50)], n_boot=2000, seed=1)
    assert 0.4 <= r["lo"] < 0.5 < r["hi"] <= 0.6 and abs(r["mean"] - 0.5) < 0.02


def test_bootstrap_deterministic():
    x = np.random.RandomState(0).randn(100)
    assert bootstrap_ci(x, seed=3) == bootstrap_ci(x, seed=3)


def test_effect_size():
    r = effect_size(1.0, np.random.RandomState(0).normal(0, 0.1, 100))
    assert r["cohens_d"] > 5 and r["percentile"] == 100.0
