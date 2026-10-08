"""Probe: does the label checker find what the injector injected?"""
import os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spine import labelqc as Q, lang, generate
log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "probe_qc.log"), "w", encoding="utf-8")
def L(s):
    log.write(str(s) + "\n"); log.flush()
try:
    q = lang.Query(pins={"maneuver.type": "lane_keep", "road.kind": "motorway"})
    sc = generate.generate(q, 7)
    clean = Q.boxes_from_scenario(sc, n_frames=40)
    L("clean boxes=%d tracks=%d" % (len(clean), len({b.track_id for b in clean})))
    rep0 = Q.quality_report(clean, mu=sc.weather.friction())
    L("clean report defects=%d kinds=%s quality=%.4f" % (rep0["defects"], rep0["defects_by_kind"], rep0["quality"]))
    allk = sorted(Q.INJECTIONS)
    corrupt, truth = Q.inject_defects(clean, allk, seed=3)
    rep1 = Q.quality_report(corrupt, mu=sc.weather.friction())
    s = Q.score_detector(Q.run_checks(corrupt, mu=sc.weather.friction()), truth)
    L("injected %d truth, detected kinds=%s" % (len(truth), s["detected_by_kind"]))
    L("per_class=%s" % s["per_class"])
    L("recall=%s missing=%s" % (s["recall"], s["missing"]))
    L("corrupt quality=%.4f defects=%d" % (rep1["quality"], rep1["defects"]))
except BaseException as e:
    L("ERR %s: %s" % (type(e).__name__, e)); L(traceback.format_exc()[-2500:])
log.close()
