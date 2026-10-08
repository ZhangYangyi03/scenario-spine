"""End-to-end CLI test: every subcommand must run and must write what it says.

Not a smoke test for its own sake. Two of these subcommands are the only way the
README's numbers are produced, so a CLI that exits 0 while writing a truncated
report would make the README unverifiable."""
import json, os, subprocess, sys
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(*args, timeout=900):
    return subprocess.run([sys.executable, "-m", "spine", *args], cwd=BASE,
                          capture_output=True, text=True, timeout=timeout)


def test_doctor_runs_and_reports_python():
    p = run("doctor")
    assert p.returncode == 0, p.stderr[-2000:]
    assert "python" in p.stdout and "torch" in p.stdout


def test_corpus_calibrate_compare_chain(tmp_path):
    p = run("corpus", "--train", "300", "--test", "150", "--workers", "2",
            "--t-end", "6", "--dt", "0.05")
    assert p.returncode == 0, p.stderr[-3000:]
    summary = json.loads(open(os.path.join(BASE, "data", "corpus_summary.json"), encoding="utf-8").read())
    assert summary["train"] == 300 and summary["test"] == 150
    assert 0.0 <= summary["collision_rate_train"] <= 1.0

    p = run("calibrate")
    assert p.returncode == 0, p.stderr[-3000:]
    cal = json.loads(open(os.path.join(BASE, "data", "calibration.json"), encoding="utf-8").read())
    w = cal["model"]["weights"]
    assert cal["constraints_hold"] is True
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert all(v >= -1e-12 for v in w.values())
    assert set(w) == set(cal["default_weights"])
    for side in ("default", "fitted"):
        assert cal[side]["auc_collision"] is not None

    p = run("compare", "--nodes", "20000")
    assert p.returncode == 0, p.stderr[-3000:]
    comp = json.loads(open(os.path.join(BASE, "data", "dispatch_compare.json"), encoding="utf-8").read())
    for key, r in comp.items():
        assert r["reference"]["floor_is_valid"], (key, r["reference"])
        assert all(r[m]["violations"] == 0 for m in ("greedy", "auction", "static_assignment"))


def test_odd_subcommand():
    p = run("odd", "--n", "60")
    assert p.returncode == 0, p.stderr[-3000:]
    # the pairwise number is the headline, so the CLI must print it by name
    assert "2-way (pairwise) coverage" in p.stdout, p.stdout[-800:]
    assert "outside the declared ODD" in p.stdout, p.stdout[-800:]
