"""The calibrator: the constraints hold, the projection is exact, fitting beats
not fitting on data where the answer is known, and both backends agree.

Three claims are checked and each has a reason:

  * the projection is exact. A calibrator that quietly violated `w >= 0, sum w = 1`
    would make the "share of difficulty" reading of a weight false, and nothing
    else in the suite would notice.
  * the fit recovers a planted weight vector. On data with a known right answer, a
    fitted model that cannot find it is broken no matter how it looks on real data.
  * the pure-Python path agrees with the numpy one. `spine calibrate` in the CLI
    must work without numpy -- that is the package's stated contract -- so a
    divergence between the two backends would be a contract violation that only
    shows up on the machine that lacks numpy.
"""

import json
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


def test_pure_projection_agrees_with_the_numpy_one():
    rng = np.random.default_rng(2)
    for _ in range(50):
        v = [float(x) for x in rng.normal(0, 3, size=7)]
        a = calibrate._project_simplex(v)
        b = calibrate._project_simplex_pure(v)
        assert max(abs(float(x) - y) for x, y in zip(a, b)) < 1e-9


def test_projection_is_the_nearest_simplex_point():
    """The definition, not a property: brute-force the minimum over the simplex by
    dense rejection sampling and require the projection to be no further away than
    the best sample."""
    rng = np.random.default_rng(1)
    for _ in range(4):
        v = rng.normal(0, 2, size=4)
        p = calibrate._project_simplex(v)
        d = float(((p - v) ** 2).sum())
        best = d
        for _ in range(20000):
            s = rng.dirichlet([1, 1, 1, 1])
            best = min(best, float(((s - v) ** 2).sum()))
        assert d <= best + 1e-6, (d, best)


def test_fit_recovers_a_planted_weight_vector():
    """Planted ground truth: components uniform, risk = C @ w_true + noise, and
    three components carrying no signal at all. The fit must put the weight where
    the signal is and leave the dead components near zero."""
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
    order = list(difficulty.COMPONENT_ORDER)
    dead = [order[i] for i in (0, 1, 3)]
    alive = [order[i] for i in (2, 5, 6)]
    assert max(w[order.index(k)] for k in dead) < 0.06, model["weights"]
    assert min(w[order.index(k)] for k in alive) > 0.15, model["weights"]
    mse_fit = float(((C @ w - y) ** 2).mean())
    mse_uniform = float(((C @ (np.ones(d) / d) - y) ** 2).mean())
    assert mse_fit < mse_uniform / 5.0


def test_pure_fit_recovers_the_same_planted_vector():
    """The same planted case, through the standard-library implementation. run on a
    smaller n because the pure path is O(iters * n * d) in Python."""
    rng = np.random.default_rng(7)
    n, d = 4000, len(difficulty.COMPONENT_ORDER)
    C = rng.random((n, d))
    w_true = np.array([0.0, 0.0, 0.25, 0.0, 0.15, 0.30, 0.30])
    y = C @ w_true + rng.normal(0, 0.01, n)
    model = calibrate.fit_weights_pure(C.tolist(), y.tolist(), lr=0.5, iters=6000)
    w = model["w_vector"]
    assert model["backend"] == "pure" and model["constraints_hold"]
    order = list(difficulty.COMPONENT_ORDER)
    assert max(w[order.index(order[i])] for i in (0, 1, 3)) < 0.06, model["weights"]
    assert min(w[order.index(order[i])] for i in (2, 5, 6)) > 0.15, model["weights"]


def test_two_backends_agree_on_the_same_input():
    """Not bit-identical -- the pure path accumulates in a different order -- but
    close enough that a published number cannot depend on which ran."""
    rng = np.random.default_rng(13)
    n, d = 3000, len(difficulty.COMPONENT_ORDER)
    C = rng.random((n, d))
    y = C @ np.array([0.05, 0.1, 0.2, 0.15, 0.1, 0.2, 0.2]) + rng.normal(0, 0.01, n)
    a = np.asarray(calibrate.fit_weights(C, y, iters=4000)["w_vector"])
    b = np.asarray(calibrate.fit_weights_pure(C.tolist(), y.tolist(), iters=4000)["w_vector"])
    assert np.abs(a - b).max() < 0.02, (a, b)


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
    C = rng.random((1000, d)).tolist()
    model = calibrate.fit_weights(C, rng.random(1000).tolist())
    s = calibrate.calibrated_score(C, model)
    assert min(s) >= 1.0 and max(s) <= 10.0
    # and it is a plain Python list, so a caller without numpy can use it
    assert isinstance(s, list)


def test_compare_reports_both_backends_and_both_sides():
    """compare() must not be able to report a fitted score that is worse than the
    default without saying so -- the caller needs both numbers. The case here has
    the default weights already right, so a compare() that always claimed an
    improvement would fail."""
    rng = np.random.default_rng(5)
    order = list(difficulty.COMPONENT_ORDER)
    d = len(order)
    Ctr = rng.random((20000, d))
    Cte = rng.random((5000, d))
    w_true = np.array([difficulty.DEFAULT_WEIGHTS[k] for k in order])
    w_true = w_true / w_true.sum()
    ytr = Ctr @ w_true + rng.normal(0, 0.005, 20000)
    yte = Cte @ w_true + rng.normal(0, 0.005, 5000)
    rep = calibrate.compare(Ctr, ytr, Cte, yte, (yte > 0.5).astype(float).tolist())
    assert set(rep) >= {"default", "fitted", "default_weights", "fitted_weights", "model", "backend"}
    assert rep["constraints_hold"]
    for k in ("mse_vs_risk_index", "spearman_risk", "auc_collision"):
        assert k in rep["default"] and k in rep["fitted"], k
    assert abs(rep["default"]["mse_vs_risk_index"] - rep["fitted"]["mse_vs_risk_index"]) < 5e-3
    json.dumps({k: v for k, v in rep.items() if k != "model"})


def test_compare_works_without_numpy(monkeypatch):
    """The contract `spine calibrate` depends on: with numpy unavailable, compare()
    still runs and still returns the same shape of report."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def blocked(name, *a, **k):
        if name == "numpy" or name.startswith("numpy."):
            raise ImportError("numpy blocked for this test")
        return real_import(name, *a, **k)

    # The target is built BEFORE numpy is blocked, because with the import hook in
    # place any use of numpy inside the test body would (correctly) explode.
    rng = np.random.default_rng(17)
    n, d = 500, len(difficulty.COMPONENT_ORDER)
    C = rng.random((n, d)).tolist()
    y = calib_target(C, rng, n, d)
    col = [1.0 if v > 0.5 else 0.0 for v in y]

    monkeypatch.setattr("builtins.__import__", blocked)
    rep = calibrate.compare(C[:400], y[:400], C[400:], y[400:], col[400:])
    assert rep["backend"] == "pure"
    assert rep["constraints_hold"]
    for side in ("default", "fitted"):
        for k in ("mse_vs_risk_index", "spearman_risk", "auc_collision"):
            assert k in rep[side], (side, k)


def calib_target(C, rng, n, d):
    """C @ w + noise, computed before numpy is blocked in the caller."""
    w = [0.05, 0.1, 0.2, 0.15, 0.1, 0.2, 0.2]
    noise = rng.normal(0, 0.01, n)
    return [sum(row[j] * w[j] for j in range(len(w))) + float(noise[i]) for i, row in enumerate(C)]
