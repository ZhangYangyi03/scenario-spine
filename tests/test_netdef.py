"""The exporter's ordering, and the reference forward pass.

A model exported with its layers in the wrong order runs, predicts, and is wrong.
So the order is asserted against a hand-built two-layer case whose answer is
computable on paper, not against whatever the exporter happens to produce."""

import pytest

torch = pytest.importorskip("torch")

from spine import netdef  # noqa: E402


def test_sequential_indices_are_non_contiguous_and_sorted():
    """nn.Sequential numbers every child; SiLU takes no parameters, so the linear
    layers are at 0, 2, 4. The exporter must sort by that index, not assume 0..n."""
    net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.SiLU(),
                              torch.nn.Linear(4, 2), torch.nn.SiLU(),
                              torch.nn.Linear(2, 1))
    arrays = netdef.state_to_arrays(net.state_dict(), 0)
    assert [a["in_features"] for a in arrays] == [3, 4, 2]
    assert [a["out_features"] for a in arrays] == [4, 2, 1]


def test_forward_arrays_matches_a_hand_computed_two_layer_network():
    arrays = [
        {"layer": 0, "out_features": 2, "in_features": 2,
         "weight": [[1.0, 2.0], [3.0, 4.0]], "bias": [0.5, -0.5]},
        {"layer": 1, "out_features": 1, "in_features": 2,
         "weight": [[1.0, 1.0]], "bias": [0.0]},
    ]
    x = [1.0, 1.0]
    h0 = 1.0 * 1 + 2.0 * 1 + 0.5
    h1 = 3.0 * 1 + 4.0 * 1 - 0.5
    s0 = h0 / (1.0 + pow(2.718281828459045, -h0))
    s1 = h1 / (1.0 + pow(2.718281828459045, -h1))
    assert netdef.forward_arrays(arrays, x) == pytest.approx(s0 + s1, rel=1e-9)


def test_forward_arrays_sigmoid_on_the_last_layer_only():
    """The last layer is a logit, not a probability: applying the activation to it
    would make the kernel disagree with the trained model in a way that only shows
    up as a slightly worse AUC, which is exactly the kind of bug worth a test."""
    arrays = [
        {"layer": 0, "out_features": 1, "in_features": 1,
         "weight": [[2.0]], "bias": [0.0]},
    ]
    # a single layer is the last layer, so the output must be the raw 2*x
    assert netdef.forward_arrays(arrays, [3.0]) == pytest.approx(6.0)


def test_netdef_describe_matches_the_torch_sequential_shape():
    net = torch.nn.Sequential(torch.nn.Linear(5, 3), torch.nn.SiLU(), torch.nn.Linear(3, 1))
    arrays = netdef.state_to_arrays(net.state_dict(), 0)
    assert arrays[0]["in_features"] == 5 and arrays[-1]["out_features"] == 1


def test_cpp_kernel_shape_constants_are_not_exceeded():
    """MAX_DIM is a shared memory bound in the .mu file. If the architecture ever
    grows past it the kernel would silently overrun, so the constant in the source
    is compared against the Python description rather than trusted."""
    import os
    import re

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(base, "musa", "predict.mu"), encoding="utf-8").read()
    m = re.search(r"#define\s+MAX_DIM\s+(\d+)", src)
    assert m, "MAX_DIM must be defined in musa/predict.mu"
    max_dim = int(m.group(1))
    in_dim = 48
    d = netdef.describe(in_dim)
    widest = max([in_dim] + [h for _, _, h in d["layers"]])
    assert widest <= max_dim, (widest, max_dim)
