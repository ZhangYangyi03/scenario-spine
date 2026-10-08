# scenario-spine

A scenario bench for automated driving that measures the four things a scenario
bench is usually asked to assume: that its scenarios are the right ones, that they
are hard enough to matter, that its labels are trustworthy, and that the fleet
running them is dispatched sanely.

Two of those turned into results worth reporting, and neither is the one I
expected when I started:

* A hand-set difficulty formula that ranks scenarios **worse than chance**
  against what the simulator actually does (AUC 0.567 on 30k held-out
  scenarios). Fitting its weights to 200k simulated scenarios lifts that to
  **0.679** and the rank correlation from 0.084 to **0.352**.
* A coverage-directed generator reaches **100% of the pairwise interaction cells**
  of the declared ODD in **37 scenarios**, where uniform sampling has not reached
  90% after **4000**.

Everything below is measured in `bench/results.json` and reproduced by
`python bench/experiment.py`.

## What is in it

    spine/lang.py        a scenario description language: a query, a parameter
                         vector and a generated scenario are the same object
    spine/generate.py    the generator: every actor placement is computed from
                         the parameters, not sampled
    spine/sim.py         a time-stepped simulator (IDM car-following, lane changes,
                         AEB) producing TTC, gap, deceleration and jerk
    spine/difficulty.py  a difficulty score with a full attribution trail
    spine/calibrate.py   fits that score's weights to simulation
    spine/features.py    a 48-dim pre-execution feature vector
    spine/dataset.py     parallel, seed-reproducible corpus construction
    spine/learn.py       the learned predictor (numpy reference + torch)
    spine/odd.py         three-valued ODD verdicts and pairwise coverage
    spine/labelqc.py     3D label quality control
    spine/fleet.py       dispatch and charging: greedy, auction, exact
    spine/assurance.py   a safety case auto-emitted from the runs
    musa/                the predictor's forward pass as a MUSA kernel

## Results

Measured on an MTT S4000 box (MUSA 3.1.0, torch_musa 1.3.0, 128 cores).
Corpus: 200,000 training scenarios from seeds 0..200000, 30,000 test scenarios
from seeds 1000000..1030000. **The split is by seed range, not by row** -- a random
row split leaks, because the generator is continuous in its parameters and two
adjacent seeds differ in the third decimal.

### 1. The difficulty score does not predict difficulty

This is the negative result the project is built around. `difficulty.py` scores a
scenario 1..10 from seven structural components with weights chosen by hand and
justified in a comment. Against the simulator's own risk index, on scenarios it
never saw:

    score                  rank corr.   AUC (collision)   MSE
    hand-set weights            0.084           0.567        0.127
    weights fitted to 200k      0.352           0.679        0.068

The hand-set score's AUC of 0.567 is barely above the 0.5 of a coin, and its rank
correlation is 0.084. Fitting the weights against simulation more than doubles the
rank correlation and moves AUC to 0.679. Two of the seven hand-chosen components --
`time_budget` (weight 0.22) and `occlusion` (0.13), together a third of the total --
are driven to **exactly zero** by the fit, which says those two terms carried no
signal that the other five did not already carry.

The fitted weights are in `bench/results.json`; the hand-set ones are kept in the
source, because the comparison is the finding. `calibrate.py` constrains the fit to
`w >= 0, sum w = 1`, so the score stays a convex combination and a weight still
reads as "share of the total difficulty".

### 2. Aimed generation versus uniform sampling

Declared ODD: `highway-pilot-v1`, 15 axes, 4 bins per axis, 1405 pairwise
interaction cells. Both arms draw 400 scenarios from the same generator over the
same grid; the only difference is whether each draw is aimed at a cell the corpus
has not covered.

    sampler      pairwise coverage   scenarios to reach 90%
    uniform                  0.493                   >4000
    aimed                    1.000                      37

100% pairwise coverage in 37 scenarios. Uniform sampling reaches 49.3% at 400 and
never reaches 90% within 4000. The number that matters is not the 1.000 -- it is
that the aimed arm needs **37 scenarios to do what the unaimed arm cannot do in
4000**, a factor of 100+ on the same generator.

Corpus throughput: 200,000 scenarios in 100.4 s (0.502 ms each) on 120 processes,
with a 11.6% collision rate and a 13.7% hard-braking rate.

### 3. What the learned predictor is actually worth

The same 48 pre-execution features, an MLP (256, 256, 128, SiLU), trained on the
MTT S4000. It is scored against the simulator on the same held-out seeds:

    predictor                       AUC    rank corr. vs risk
    MLP, collision target         0.9999              --
    MLP, risk-index target        0.9993            0.997
    fitted analytic score         0.679             0.352
    hand-set analytic score       0.567             0.084

The honest reading: this is **near-perfect by construction and it is not a
surprising result**. The features include the physics the simulator implements
(stopping distance over sight distance, grip fraction, headway in seconds), so a
high-capacity model recovers an analytic relationship it was handed. Its value is
as a *surrogate*, and as a surrogate it is worth a great deal: it scores 30,000
scenarios in 1.2 ms against 19.3 s to simulate them, a factor of about 16,000.

What that buys, measured on the 30,000 held-out scenarios of which 3,364 collide:

    screen top 1% (300 scenarios)     found 300 collisions   precision 1.000
    screen top 5% (1500 scenarios)    found 1500 collisions  precision 1.000
    same split, hand-set score (top 1%):   23 collisions     precision 0.077

The top 1% by the learned score contains **300 of 300 real collisions** -- every
scenario selected collides when simulated, an 8.9x lift over the base rate. The
hand-set score's top 1% contains 23 collisions, below the base rate: sampling at
random would have done about as well. The top 5% recovers 44.6% of all collisions
in the corpus.

So the practical claim is not "a neural network predicts collisions". It is: **a
scenario corpus 50x smaller, chosen by the surrogate, contains more real
collisions than the full corpus filtered by the hand-set score**, and the
simulation budget goes to candidates that are worth simulating.

### 4. The GPU side, in its own words

The bottleneck is generation, and generation is CPU work -- a Python time-stepped
loop with nothing to vectorise. The GPU trains and runs the predictor. A
hand-written MUSA kernel (`musa/predict.mu`) reimplements the forward pass and
**agrees with the numpy reference to 6.4e-5 on 4000 held-out rows**, rank
correlation exactly 1.0. That kernel exists for verification, not speed; the
README does not claim a speedup it does not have. Details, including the two
warnings that cost real time, are in `docs/GPUS.md`.

## Three-valued logic, not booleans

An ODD verdict per axis is `SAT` / `VIOLATED` / `UNKNOWN`, where `UNKNOWN` means
the scenario does not carry that axis at all. Collapsing that to a boolean is how a
coverage report ends up claiming an axis was tested when it was never populated.
`spine/odd.py` keeps the third value and reports the fraction of axes in each
state; the tests assert that a partial scenario yields `UNKNOWN` rather than a
silent pass.

Full-product coverage is reported as a number
(169,869,312 cells for this ODD, coverage ~1e-5) precisely so that the fact it is
out of reach is a measured quantity and not a caveat in prose.

## Install and run

No dependencies are needed for the core: the generator, simulator, difficulty
score, coverage measure, label QC and dispatcher are standard-library Python.
numpy and torch are optional and only the tests that need them import them.

    python -m pytest                      # 53 tests, stdlib-only paths included
    python -m spine doctor                # what this machine can run
    python -m spine corpus --train 20000  # build a corpus
    python -m spine calibrate             # fit the difficulty weights
    python -m spine compare               # dispatch: greedy vs auction vs exact
    python -m spine odd --n 200           # ODD coverage report

The dispatcher (`python -m spine compare`) is checked against a branch-and-bound
optimum where it can be proven, and against an independent lower bound where it
cannot; every method is asserted to produce zero constraint violations. The safety
case generator (`spine/assurance.py`) is fully checked but its *content* is not
validated against a domain expert's review, and it says so.

## Honest limits

* The simulator is a model, not the world. The risk index the predictor is trained
  against is *this simulator's* risk index, so "predicts collisions" means "predicts
  this simulator's collisions". The calibration result is a statement about the
  formula's internal consistency, and the negative finding above is robust to it
  (both scores are judged by the same simulator), but the learned model's AUC
  should not be read as a real-world collision rate.
* The learned model wins partly because the features contain the physics. That is
  stated rather than hidden, and the analytic baseline is the honest comparison.
* Coverage is over the declared ODD's axes and a 4-bin discretisation. It is not a
  claim about the ODD's operational completeness.
* The hand-written kernel is a correctness artefact, not a performance one. There
  is no kernel speedup claim in this README.

## Licence

Apache-2.0. Citation metadata in `CITATION.cff`.
