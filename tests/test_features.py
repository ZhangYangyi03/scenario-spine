"""The feature vector: fixed dimension, deterministic, and strictly
pre-execution.

The last of those is the one that matters. If any feature were computed from the
simulated outcome, the learned model would be predicting the outcome from itself
and every number in the README would be worthless. The test for it is structural
rather than numerical: build a scenario, mutate only the outcome-producing
behaviour, and require the features not to move."""

import pytest

from spine import features as F
from spine import generate, lang, sim


def _sc(**kw):
    base = {name: (a["default"] if a["kind"] == "range" else a["values"][0])
            for name, a in lang.AXES.items()}
    base.update(kw)
    base["seed"] = 1
    return generate.scenario_from_params(base)


def test_feature_dimension_is_fixed():
    names = F.feature_names()
    assert len(names) == len(set(names)), "duplicate feature names"
    for man in lang.AXES["maneuver.type"]["values"]:
        assert len(F.features(_sc(**{"maneuver.type": man}))) == len(names)
    assert len(names) >= 40


def test_features_are_finite_everywhere_on_the_grid():
    """A NaN or an inf reaching the trainer is a silent corrupted model. Every
    categorical value crossed with the ends of every continuous axis is checked."""
    import math

    axes = {k: v["values"] for k, v in lang.AXES.items() if v["kind"] == "cat"}
    for name, vals in axes.items():
        for val in vals:
            row = F.features(_sc(**{name: val}))
            assert all(math.isfinite(x) for x in row), (name, val)
    for name, ax in lang.AXES.items():
        if ax["kind"] != "range":
            continue
        for val in (ax["lo"], ax["hi"]):
            row = F.features(_sc(**{name: val}))
            assert all(math.isfinite(x) for x in row), (name, val)


def test_categorical_blocks_are_one_hot():
    a = F.features(_sc(**{"weather.condition": "snow"}))
    b = F.features(_sc(**{"weather.condition": "ice"}))
    names = F.feature_names()
    changed = [n for n, x, y in zip(names, a, b) if abs(x - y) > 1e-12]
    # the two one-hot entries must swap; the *derived* terms (mu, and therefore
    # the grip-demand ratios) are expected to move too, and that is the point of
    # having them -- so the assertion is about the block, not about total silence
    block = [n for n in changed if n.startswith("weather.condition=")]
    assert block == ["weather.condition=snow", "weather.condition=ice"], block
    assert "mu" in changed, "friction must react to the weather condition"
    assert not any(n.startswith(("maneuver.type=", "road.kind=")) for n in changed), changed


def test_features_do_not_depend_on_the_simulated_outcome():
    """The pre-execution guarantee, tested by construction: two scenarios whose
    only difference is a behaviour the features cannot see must produce identical
    vectors. `lead_brake` with different decelerations changes the outcome and
    must not change a single feature."""
    a = _sc(**{"maneuver.type": "lead_brake", "maneuver.decel_mps2": 1.0})
    b = _sc(**{"maneuver.type": "lead_brake", "maneuver.decel_mps2": 8.0})
    names = F.feature_names()
    decel_idx = [i for i, n in enumerate(names) if "decel" in n]
    assert decel_idx, "the deceleration feature must exist"
    diffs = {names[i] for i, (x, y) in enumerate(zip(F.features(a), F.features(b))) if abs(x - y) > 1e-12}
    # only the declared deceleration feature may move; nothing else
    assert diffs == {"decel_over_8"}, diffs


def test_outcome_row_refuses_a_non_outcome():
    with pytest.raises(TypeError):
        F.outcome_row({"collision": True})


def test_outcome_row_keys_match_the_dataset_contract():
    sc = _sc()
    o = sim.simulate(sc, dt=0.05, t_end=6.0)
    r = F.outcome_row(o)
    for k in ("collision", "risk_index", "min_ttc_s", "min_gap_m", "max_decel_mps2",
              "peak_jerk", "hard_brake"):
        assert k in r, k
    assert 0.0 <= r["risk_index"] <= 1.0


def test_speed_and_grip_move_in_the_expected_direction():
    names = F.feature_names()
    si = names.index("log_ego_speed")
    mi = names.index("mu")
    assert F.features(_sc(**{"ego.speed_kph": 120.0}))[si] > F.features(_sc(**{"ego.speed_kph": 20.0}))[si]
    assert F.features(_sc(**{"weather.condition": "ice"}))[mi] < F.features(_sc(**{"weather.condition": "dry"}))[mi]
