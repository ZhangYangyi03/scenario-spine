"""The README's numbers have to be the benchmark's numbers.

A README drifts from the code that produced it, and the drift is invisible because
nobody recomputes a table by hand. So the table is checked: every headline figure in
README.md is read back out of bench/results.json and compared. If a number in the
README is not in the results file, or disagrees with it, this fails.

The test skips when the results file is absent, which is the case on a fresh clone
before `python bench/experiment.py` has been run -- and it says so in the skip
message rather than passing quietly."""

import json
import os

import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "bench", "results.json")
README = os.path.join(BASE, "README.md")


@pytest.fixture(scope="module")
def results():
    if not os.path.exists(RESULTS):
        pytest.skip("bench/results.json not present: run python bench/experiment.py first")
    with open(RESULTS, encoding="utf-8") as f:
        return json.load(f)


def test_calibration_numbers_in_the_readme_are_the_measured_ones(results):
    cal = results["calibration"]
    text = open(README, encoding="utf-8").read()
    for key in ("spearman_risk", "auc_collision", "mse_vs_risk_index"):
        for side in ("default", "fitted"):
            v = cal[side][key]
            assert v is not None
            # three significant digits, which is how the README quotes them
            shown = f"{v:.3f}".rstrip("0").rstrip(".")
            assert shown in text, f"{side}.{key} = {v} ({shown}) is not in the README"


def test_fitted_beats_default_on_the_held_out_seeds(results):
    cal = results["calibration"]
    assert cal["constraints_hold"] is True
    assert sum(cal["fitted_weights"].values()) == pytest.approx(1.0, abs=1e-6)
    assert all(v >= -1e-12 for v in cal["fitted_weights"].values())
    assert cal["fitted"]["auc_collision"] > cal["default"]["auc_collision"] + 0.05
    assert cal["fitted"]["spearman_risk"] > cal["default"]["spearman_risk"] + 0.1


def test_coverage_numbers_in_the_readme_are_the_measured_ones(results):
    cov = results["coverage"]
    assert cov["aimed_at_equal_size"] > cov["unaimed_at_equal_size"]
    # The unaimed arm's number is a *string* when it did not reach the target
    # within its cap (">4000"), which is deliberate: "never reached" and "reached
    # at 4001" are different claims and the results file must not conflate them.
    aimed = cov["scenarios_to_90pct_aimed"]
    unaimed = cov["scenarios_to_90pct_unaimed"]
    assert isinstance(aimed, int) and aimed > 0
    if isinstance(unaimed, str):
        assert unaimed.startswith(">")
        unaimed_n = int(unaimed[1:])
    else:
        unaimed_n = int(unaimed)
    assert aimed < unaimed_n, (aimed, unaimed)
    text = open(README, encoding="utf-8").read()
    assert f"{cov['unaimed_at_equal_size']:.3f}" in text
    assert str(aimed) in text and unaimed in text


def test_learned_metrics_are_reported_with_their_device(results):
    led = results["learned"]
    if led.get("skipped"):
        pytest.skip("learned half was skipped in this run (SPINE_SKIP_LEARNED=1)")
    assert led["device"] in ("musa", "cuda", "cpu")
    assert led["mlp_collision"]["auc_collision"] is not None
    assert led["mlp_risk"]["spearman_risk"] is not None
    # a reliability table with a near-empty top decile would mean the AUC came
    # from somewhere other than the ranking it is supposed to describe
    table = led["reliability_deciles"]
    assert len(table) == 10
    assert table[0]["measured_collision_rate"] > 0.9


def test_screening_numbers_agree_with_the_table(results):
    path = os.path.join(BASE, "bench", "screening.json")
    if not os.path.exists(path):
        pytest.skip("bench/screening.json not present")
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    text = open(README, encoding="utf-8").read()
    top1 = s["learned_mlp"]["top_1pct"]
    assert top1["precision"] == 1.0
    assert str(top1["collisions_found"]) in text
