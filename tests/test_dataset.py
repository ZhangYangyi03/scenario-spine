"""The dataset: reproducible, split by seed, and rebuildable.

The rebuild test is the one worth explaining. add_physics does not store the
scenario, it reconstructs it from the parameter vector. That is a claim: the
generator is a pure function of its parameters. If it ever stopped being one --
because someone reached for a global random source, or a dict iteration order --
the physics column would silently start describing a different scenario from the
one that was simulated, and every comparison between the learned model and the
analytic score would be between two different corpora. So the claim is tested by
re-simulating."""

import os

import pytest

from spine import dataset as D
from spine import generate, lang, sim


def test_build_is_deterministic():
    a = D.build(n=40, seed0=0, t_end=6.0)
    b = D.build(n=40, seed0=0, t_end=6.0)
    assert [r["scenario"] for r in a] == [r["scenario"] for r in b]
    assert [r["y"] for r in a] == [r["y"] for r in b]


def test_parallel_and_serial_agree():
    """A work-stealing pool would not give this, which is why the pool is over
    contiguous seed blocks: the dataset must not depend on the worker count."""
    s = D.build(n=96, seed0=0, workers=1, t_end=6.0)
    p = D.build(n=96, seed0=0, workers=4, t_end=6.0)
    assert [r["scenario"] for r in s] == [r["scenario"] for r in p]
    assert [r["y"]["risk_index"] for r in s] == [r["y"]["risk_index"] for r in p]


def test_rebuilding_from_params_reproduces_the_simulated_outcome():
    rows = D.build(n=25, seed0=500, t_end=8.0)
    for r in rows:
        sc = generate.scenario_from_params(r["params"], r["scenario"])
        o = sim.simulate(sc, dt=0.02, t_end=8.0)
        assert o.collision == bool(r["y"]["collision"])
        assert abs(o.risk_index() - r["y"]["risk_index"]) < 1e-9


def test_physics_column_is_attached_and_on_scale():
    rows = D.build(n=25, seed0=0, physics=True, t_end=6.0)
    assert all(r["physics"] is not None for r in rows)
    assert all(1.0 <= r["physics"] <= 10.0 for r in rows)


def test_split_is_by_seed_and_is_a_partition():
    rows = D.build(n=60, seed0=0, t_end=6.0)
    tr, te = D.split_by_seed(rows, 30, 60)
    assert len(tr) + len(te) == len(rows)
    assert all(not (30 <= r["seed"] < 60) for r in tr)
    assert all(30 <= r["seed"] < 60 for r in te)
    assert not (set(r["seed"] for r in tr) & set(r["seed"] for r in te))


def test_jsonl_roundtrip_compressed_and_plain(tmp_path):
    rows = D.build(n=20, seed0=0, t_end=6.0)
    for comp in (True, False):
        path = str(tmp_path / f"rows{int(comp)}.jsonl")
        D.write_jsonl(rows, path, compress=comp)
        back = D.read_jsonl(path)
        assert len(back) == len(rows)
        assert back[0]["params"] == rows[0]["params"]
        assert abs(back[0]["y"]["risk_index"] - rows[0]["y"]["risk_index"]) < 1e-12


def test_a_seed_never_collides_across_blocks():
    """Blocks are laid out by offset; an off-by-one would make the last scenario of
    one block the first of the next, inflating the effective corpus size."""
    a = D.build(n=50, seed0=0, workers=1, t_end=4.0)
    b = D.build(n=50, seed0=50, workers=1, t_end=4.0)
    assert not (set(r["seed"] for r in a) & set(r["seed"] for r in b))


def test_spearman_and_auc_agree_with_known_answers():
    assert D.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert D.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # a perfect ranker of a balanced binary target scores AUC 1
    m = D.metrics_pure([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["auc"] == pytest.approx(1.0)
    m = D.metrics_pure([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
    assert m["auc"] == pytest.approx(0.0)
    # all-tied predictions must give exactly 0.5, not depend on tie order
    m = D.metrics_pure([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5])
    assert m["auc"] == pytest.approx(0.5)
