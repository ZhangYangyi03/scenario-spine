"""Run the whole study and write its artefacts. The command the README quotes.

    python tools/run_study.py --budget 90 --out out

Kept as a file in the repository rather than a heredoc, because a one-liner that
has to be retyped is a one-liner that gets retyped wrong.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spine import odd as oddmod, study  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the scenario-spine end-to-end study.")
    ap.add_argument("--budget", type=int, default=90, help="max scenarios to generate")
    ap.add_argument("--bins", type=int, default=4, help="bins per continuous ODD axis")
    ap.add_argument("--strategy", default="gap", choices=["gap", "random"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fleet-scale", type=float, default=1.5)
    ap.add_argument("--odd", default="highway", help="highway | urban")
    ap.add_argument("--out", default="out", help="artefact directory")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    odd = oddmod.declared_odd_highway() if a.odd == "highway" else oddmod.declared_odd_urban()
    art = study.run_study(odd, budget=a.budget, bins_per_axis=a.bins, strategy=a.strategy,
                          seed=a.seed, fleet_scale=a.fleet_scale)
    paths = study.write_artifacts(art, a.out)
    clean = {k: v for k, v in art.items() if not k.startswith("_")}
    if not a.quiet:
        print(json.dumps(clean, indent=2))
    print("wrote:", ", ".join(sorted(os.path.basename(p) for p in paths.values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
