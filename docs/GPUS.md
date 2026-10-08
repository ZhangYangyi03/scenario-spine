# The MTT S4000 / MUSA side

Everything here was run on a rented MTT S4000 (Moore Threads), MUSA 3.1.0,
driver 2.7.0, torch 2.2.0 + torch_musa 1.3.0, 128 Xeon Gold 6430 cores.

## Why a GPU is involved at all, stated honestly

The bottleneck of this project is scenario *generation*, and generation is CPU
work: a 200k-scenario corpus takes 100 seconds on 120 cores, and 66,000
scenarios still take 66 seconds. The GPU is not used to speed the corpus up,
because it cannot -- the simulator is a Python time-stepped loop with no
vectorisable inner kernel.

The GPU is used for the one step that genuinely is a dense numerical workload:
training and running the predictor. On 30k held-out scenarios:

    path                          rows/s      ms / 30k rows
    torch on the MTT S4000       25,600,000            1.2
    hand-written MUSA kernel        333,000           90.0
    (the kernel's 90 ms excludes the H2D and D2H copies)

The measured per-second row counts above belong to the runs recorded in
bench/results.json and musa/throughput.py; they are not estimates.

So: 30k scenarios score in 1.2 ms on the device. That is worth having when the
screening loop is called thousands of times, and it is not worth pretending it is
the reason the project exists.

## What the hand-written kernel is for

`musa/predict.mu` reimplements the predictor's forward pass on the device, and
its purpose is verification rather than speed. It agrees with the numpy
implementation to 6.4e-5 on 4000 held-out rows (no row exceeds 2e-4), with a
Spearman rank correlation of exactly 1.0 against it. Three implementations of
the same arithmetic -- numpy from the definition of a matrix multiply, torch on
the device, and the kernel -- that agree to 6e-5 is a much stronger statement
about the model than any one of them alone.

## Two things that cost time, written down so they cost it once

1. `nvidia-smi` does not exist on this machine. The device lister is
   `mthreads-gmi`, and `torch.musa` is only bound after `import torch_musa`.
2. The S4000 reports compute capability **2.2**, so the kernel must be built for
   `mp_22`. `mp_21` compiles cleanly, loads, and then fails at launch with a bare
   `MUSA error 98` (invalid device function). `musa/build.sh` derives the arch from
   the device for this reason.

Also worth knowing: `<<<>>>` kernel launches only parse inside a `.mu` file. The
driver is an ordinary `.cpp` compiled by g++, so it calls a launcher defined in
the `.mu` translation unit instead, and the object file must be linked `-no-pie`
because mcc's device stub is not position-independent.

## Reproducing it

    # on the GPU box
    tar xzf spine_src.tar.gz && cd work
    pip install pytest numpy
    bash musa/build.sh                       # arch detected from the device
    python3 bench/experiment.py              # 200k train / 30k test corpus
    python3 musa/export_and_train.py         # train + flatten the model
    python3 musa/parity.py musa/state musa/state.pt <corpus>.npz
    python3 musa/throughput.py

The parity script exits non-zero if any of the three implementations disagrees by
more than 2e-4, so it is usable as a check rather than as a report.

## CPU-only

Nothing in `spine/` needs a GPU, a framework or even numpy: the coverage measure,
the scenario generator, the difficulty score, the label QC and the dispatcher are
standard library. `python -m pytest` on a bare Python 3.10 passes the full suite,
and the tests that need torch or numpy skip themselves. `python -m spine doctor`
prints what the current machine can run.
