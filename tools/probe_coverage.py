
import sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "probe.log"), "w", encoding="utf-8")
def L(s):
    log.write(str(s) + "\n"); log.flush()
from spine import odd as O, study
try:
    odd = O.declared_odd_highway()
    for strat in ("gap", "random"):
        corpus, run = study.build_corpus(odd, budget=120, strategy=strat)
        rep = run["report"]
        L("%-6s scenarios=%3d order2=%.4f order1=%.4f cells=%d/%d missing=%d" % (
            strat, len(corpus), rep["order2_coverage"], rep["order1_coverage"],
            rep["interaction_cells_hit"], rep["interaction_cells_total"], rep["interaction_cells_missing"]))
except BaseException as e:
    L("ERR %s: %s" % (type(e).__name__, e))
    L(traceback.format_exc()[-2000:])
log.close()
