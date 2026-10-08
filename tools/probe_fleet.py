"""Probe: the dispatch comparison, with the two references separated."""
import os, sys, traceback, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spine import fleet
log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "probe_fleet.log"), "w", encoding="utf-8")
def L(s):
    log.write(str(s) + "\n"); log.flush()
try:
    for nv, nt in [(6, 6), (8, 9), (12, 14), (16, 20)]:
        veh, trips, st = fleet.instance(nv, nt, seed=nv * 100 + nt)
        t0 = time.time()
        res = fleet.compare(veh, trips, exact_nodes=200000)
        r = res["reference"]
        L("--- nv=%d nt=%d (%.2fs)  opt=%.1f proven=%s nodes=%s served=%d floor=%.1f (ratio %.2f) static=%.1f" % (
            nv, nt, time.time()-t0, r["sequential_optimum"], r["proven_optimal"], r["bnb_nodes"],
            r["served_by_optimum"], r["independent_floor"], r["floor_ratio"], r["static_formulation_objective"]))
        for name in ("greedy", "auction", "static_assignment", "sequential_opt"):
            x = res[name]
            L("     %-18s served=%2d obj=%10.2f gapOpt=%8s floorGap=%8s viol=%d %s" % (
                name, x["served"], x["objective"], x.get("gap_vs_opt_pct"), x.get("gap_vs_floor_pct"),
                x["violations"], x["note"][:44]))
        assert r["floor_is_valid"], "INVALID FLOOR"
except BaseException as e:
    L("ERR %s: %s" % (type(e).__name__, e)); L(traceback.format_exc()[-2000:])
log.close()
