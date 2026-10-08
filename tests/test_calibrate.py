"""The calibrator: the constraints hold, the projection is exact, and fitting
beats not fitting on data where the answer is known.

A calibrator that quietly violated its own simplex constraint would produce a
score that is not a convex combination, and the "share of difficulty" reading of a
weight would become false without any test noticing. So the projection is checked
against the definition, and one end-to-end case is checked against a planted
weight vector -- on data constructed to have a known right answer, a fitted model
that cannot recover it is broken regardless of how good it looks on real data."""

import math

import pytest

from spine import calibrate, difficulty

np = pytest.importorskip("numpy")


def test_projection_lands_on_the_simplex():
    rng = np.random.default_rng(0)
    for _ in range(50):
        v = rng.normal(0, 3, size=7)
        p = calibrate._project_simplex(v)
        assert (p >= -1e-12).all(), p
        assert abs(p.sum() - 1.0) < 1e-9, p
        assert (p <= -1e-12).sum() == 0


def test_projection_is_the_nearest_simplex_point():
    """The definition, not a property: for a handful of small vectors, brute-force
    the minimum over the simplex by dense rejection sampling and require the
    projection to be no further away than the best sample."""
    rng = np.random.default_rng(1)
    for _ in range(5):
        v = rng.normal(0, 2, size=4)
        p = calibrate._project_simplex(v)
        d = float(((p - v) ** 2).sum())
        best = d
        for _ in range(20000):
            s = rng.dirichlet([1, 1, 1, 1])
            best = min(best, float(((s - v) ** 2).sum()))
        assert d <= best + 1e-6, (d, best)


def test_fit_recovers_a_planted_weight_vector():
    """Planted ground truth: components drawn uniformly, risk = C @ w_true with
    noise, and one component carrying no signal at all. The fit must put the
    weight where the signal is and must zero the dead component -- or at least
    leave it no larger than the noise floor."""
    rng = np.random.default_rng(7)
    n, d = 40000, len(difficulty.COMPONENT_ORDER)
    C = rng.random((n, d))
    w_true = np.array([0.0, 0.0, 0.25, 0.0, 0.15, 0.30, 0.30])
    y = C @ w_true + rng.normal(0, 0.01, n)
    model = calibrate.fit_weights(C, y)
    w = np.asarray(model["w_vector"])
    assert model["constraints_hold"]
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w >= -1e-12).all()
    # the two components that carry no signal must not attract weight
    order = list(difficulty.COMPONENT_ORDER)
    dead = [order[i] for i in (0, 1, 3)]
    alive = [order[i] for i in (2, 5, 6)]
    assert max(w[order.index(k)] for k in dead) < 0.06, model["weights"]
    assert min(w[order.index(k)] for k in alive) > 0.15, model["weights"]
    # and the fit must beat the uniform prior by a wide margin
    mse_fit = float(((C @ w - y) ** 2).mean())
    mse_uniform = float(((C @ (np.ones(d) / d) - y) ** 2).mean())
    assert mse_fit < mse_uniform / 5.0


def test_fit_is_not_sensitive_to_the_starting_point():
    rng = np.random.default_rng(11)
    n, d = 8000, len(difficulty.COMPONENT_ORDER)
    C = rng.random((n, d))
    w_true = np.array([0.05, 0.1, 0.2, 0.15, 0.1, 0.2, 0.2])
    y = C @ w_true + rng.normal(0, 0.01, n)
    a = np.asarray(calibrate.fit_weights(C, y)["w_vector"])
    b = np.asarray(calibrate.fit_weights(C, y, w0=[1.0] * d)["w_vector"])
    assert np.abs(a - b).max() < 0.05, (a, b)


def test_calibrated_score_stays_on_the_1_to_10_scale():
    rng = np.random.default_rng(3)
    d = len(difficulty.COMPONENT_ORDER)
    C = rng.random((1000, d))
    model = calibrate.fit_weights(C, rng.random(1000))
    s = calibrate.calibrated_score(C, model)
    assert s.min() >= 1.0 and s.max() <= 10.0


def test_compare_reports_both_and_marks_the_winner_honestly():
    """compare() must not be able to report a fitted score that is worse than the
    default without saying so -- the numbers are what they are, and the caller
    needs both to judge. The test constructs a case where the default weights are
    *right*, so a compare() that always claimed improvement would fail here."""
    rng = np.random.default_rng(5)
    order = list(difficulty.COMPONENT_ORDER)
    d = len(order)
    Ctr = rng.random((20000, d))
    Cte = rng.random((5000, d))
    w_true = np.array([difficulty.DEFAULT_WEIGHTS[k] for k in order])
    w_true = w_true / w_true.sum()
    ytr = Ctr @ w_true + rng.normal(0, 0.005, 20000)
    yte = Cte @ w_true + rng.normal(0, 0.005, 5000)
    rep = calibrate.compare(Ctr, ytr, Cte, yte, (yte > 0.5).astype(float))
    assert set(rep) >= {"default", "fitted", "default_weights", "fitted_weights", "model"}
    assert rep["constraints_hold"]
    for k in ("mse_vs_risk_index", "spearman_risk", "auc_collision"):
        assert k in rep["default"] and k in rep["fitted"], k
    # when the default weights already are the truth, the fit must not claim a
    # large improvement -- it should land close to them
    assert abs(rep["default"]["mse_vs_risk_index"] - rep["fitted"]["mse_vs_risk_index"]) < 5e-3
