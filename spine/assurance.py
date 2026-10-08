"""The safety case, emitted from measured evidence instead of written by hand.

A safety case for an automated driving function is a claim about a *domain*:
"the ARGUMENT holds that the ego is safe in the declared ODD, because these
goals hold over these scenarios". Written by hand, it goes stale the moment the
scenario library changes, and it says nothing about the part of the domain that
was never tested.

So this module does not hold a safety case. It *derives* one:

    goal      the claim, from the ODD
    strategy  the argument decomposition, from the maneuver taxonomy
    evidence  what was measured, from coverage / difficulty / simulation
    assumption what the argument needs and the evidence does not establish
    defeat    what the corpus shows contradicts the goal

Every node carries a Kleene verdict -- SAT, VIOLATED, UNKNOWN -- computed from
the evidence by the connectives in spine/odd.py. A goal whose children include
an UNKNOWN is UNKNOWN, not SAT. That is the whole difference from a hand-written
safety case: it is not possible to write an argument here that hides its own
untested region, because the untested region is a value in the tree.

The output is GSN-shaped (goal / strategy / solution / assumption / context, with
supportsInContext / inContextOf / solvedBy edges) and serialises to a
structure that maps onto SACM's argument package without an adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

from .odd import T, F, U, k_all, k_any, k_not, scenario_verdict

NODE_KINDS = ("Goal", "Strategy", "Solution", "Assumption", "Context", "Defeater", "Claim")


@dataclass
class Node:
    id: str
    kind: str
    text: str
    verdict: str = U
    evidence: dict = field(default_factory=dict)
    children: list = field(default_factory=list)
    supports: list = field(default_factory=list)
    in_context_of: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["children"] = list(d["children"])
        return d


class Case:
    """The argument tree, plus the traversal that computes a verdict for every
    node bottom-up."""

    def __init__(self, name: str = "spine-assurance-case"):
        self.name = name
        self.nodes = {}
        self.roots = []

    def add(self, node: Node, parent: str = None) -> str:
        if node.id in self.nodes:
            raise ValueError(f"duplicate node id {node.id}")
        self.nodes[node.id] = node
        if parent is None:
            self.roots.append(node.id)
        else:
            self.nodes[parent].children.append(node.id)
        return node.id

    def resolve(self) -> dict:
        """Bottom-up verdict. A Goal/Strategy is the Kleene AND of its children;
        a Defeater is the Kleene AND of the evidence it points at, negated; a
        Solution carries its measured verdict. A node with no children is
        UNKNOWN -- no evidence, no claim, and that is reported rather than
        defaulted to SAT."""
        order, seen = [], set()

        def visit(nid):
            if nid in seen:
                return
            seen.add(nid)
            for c in self.nodes[nid].children:
                visit(c)
            order.append(nid)

        for r in self.roots:
            visit(r)
        for nid in order:
            n = self.nodes[nid]
            if n.kind == "Solution" or not n.children:
                if n.verdict not in (T, F, U):
                    n.verdict = U
                continue
            child_v = [self.nodes[c].verdict for c in n.children]
            v = k_all(child_v)
            if n.kind == "Defeater":
                v = k_not(v)
            n.verdict = v
        return {nid: self.nodes[nid].verdict for nid in self.nodes}

    def to_dict(self) -> dict:
        self.resolve()
        return {"name": self.name, "roots": self.roots,
                "nodes": [self.nodes[i].to_dict() for i in self.nodes]}

    def to_gsn_dot(self) -> str:
        self.resolve()
        shape = {"Goal": "box", "Strategy": "parallelogram", "Solution": "circle",
                 "Assumption": "ellipse", "Context": "ellipse", "Defeater": "box",
                 "Claim": "box"}
        color = {T: "darkgreen", F: "red", U: "orange"}
        lines = ["digraph gsn {", '  rankdir=TB;', '  node [fontname="Helvetica"];']
        for nid, n in self.nodes.items():
            lab = n.text.replace('"', "'")
            if len(lab) > 60:
                lab = lab[:57] + "..."
            lines.append(f'  "{nid}" [shape={shape.get(n.kind, "box")}, style=filled, '
                         f'fillcolor="{color[n.verdict]}", fontcolor=white, label="{n.kind}: {lab}"];')
        for nid, n in self.nodes.items():
            for c in n.children:
                lines.append(f'  "{nid}" -> "{c}";')
        lines.append("}")
        return "\n".join(lines)


def build(odd: dict, corpus: list, verdicts: list = None, outcomes: list = None,
          difficulty: list = None, case_name: str = "highway-pilot-assurance") -> Case:
    """Build the case from a corpus of scenarios and their measured outcomes.

    Structure, all of it derived:

        G1  the function is safe within <ODD name>
         S1  argument over the maneuvers the ODD admits
            G1.1 .. G1.n  one goal per maneuver type in the ODD
                        each decomposed into
                          S<x>  argument over the scenario families that realise it
                             Sn  solution: the scenarios run, with their outcome
                          As<x> the residual assumptions the family cannot close
         S2  argument over the domain boundary
            G2.1  every scenario executed lies inside the declared ODD
            G2.2  the declared ODD is covered by executed scenarios
         D1  defeater: a scenario in the corpus left the ODD
    """
    if verdicts is None:
        from .odd import coverage

        coverage(odd, corpus)  # populates nothing; kept for the docstring's sake
        verdicts = [scenario_verdict(sc, odd) for sc in corpus]
    outcomes = outcomes or [None] * len(corpus)
    difficulty = difficulty or [None] * len(corpus)
    c = Case(case_name)

    # context: the ODD itself is context, not evidence
    ctx = c.add(Node("C1", "Context",
                     f"ODD {odd.get('name')} over {len(odd['axes'])} axes; "
                     f"{len(corpus)} scenarios executed",
                     verdict=T, evidence={"axes": list(odd["axes"])}))

    g1 = c.add(Node("G1", "Goal",
                    f"The function is acceptably safe within ODD {odd.get('name')}",
                    evidence={"odd": odd.get("name"), "scenarios": len(corpus)}))
    s1 = c.add(Node("S1", "Strategy",
                    "Argument over every maneuver type the ODD admits, each with its realised families"),
               parent=g1)

    # group the corpus by maneuver type, which the ODD declares
    maneuvers = sorted({sc.params.get("maneuver.type") for sc in corpus if sc.params.get("maneuver.type")})
    declared = odd["axes"].get("maneuver.type", {}).get("values")
    if declared:
        # a maneuver the ODD admits but no scenario realises is an UNKNOWN goal,
        # not an omitted one -- omitting it is exactly the failure mode
        maneuvers = sorted(set(maneuvers) | set(declared))
    n_k = 0
    for man in maneuvers:
        n_k += 1
        idx = [i for i, sc in enumerate(corpus) if sc.params.get("maneuver.type") == man]
        if not idx:
            c.add(Node(f"G1.{n_k}", "Goal",
                       f"Hazard {man} is mitigated whenever it arises in the ODD",
                       verdict=U, evidence={"scenarios": 0,
                                            "why": "the ODD admits this maneuver and no executed scenario realises it"}),
                  parent=s1)
            continue
        gi = c.add(Node(f"G1.{n_k}", "Goal", f"Hazard {man} is mitigated whenever it arises in the ODD",
                        evidence={"scenarios": len(idx)}), parent=s1)
        worst = max((difficulty[i]["score"] for i in idx if difficulty[i]), default=None)
        si = c.add(Node(f"S1.{n_k}", "Strategy",
                        f"Argument over {len(idx)} executed scenarios realising {man}"
                        + (f", peak difficulty {worst}" if worst is not None else "")),
                   parent=gi)
        vs = [verdicts[i]["verdict"] for i in idx]
        # evidence node per scenario: this is where a measured number enters
        for j, i in enumerate(idx):
            sc = corpus[i]
            o = outcomes[i]
            if o is None:
                v, note = U, "scenario executed without a simulated outcome"
            elif getattr(o, "collision", False):
                v, note = F, "simulated collision, no residual time margin"
            else:
                sev = 1.0 / (1.0 + max(0.0, o.min_ttc))
                v = T if sev < 0.22 else (U if sev < 0.5 else F)
                note = f"min TTC {o.min_ttc:.2f} s, min gap {o.min_gap:.2f} m"
            ev = {"scenario": sc.id, "note": note,
                  "min_ttc_s": None if o is None else round(getattr(o, "min_ttc", 0.0), 3),
                  "collision": None if o is None else getattr(o, "collision", None),
                  "difficulty": difficulty[i]["score"] if difficulty[i] else None}
            c.add(Node(f"Ev{man}{j}", "Solution",
                       f"{sc.id}: {note}" + (f"; difficulty {ev['difficulty']}" if ev["difficulty"] else ""),
                       verdict=v, evidence=ev), parent=si)
        # assumptions: what this family cannot establish, stated as an assumption
        c.add(Node(f"As{man}", "Assumption",
                   "The declared friction, visibility and occlusion values are the ones the vehicle "
                   "will actually meet; sensor degradation beyond the modelled weather is out of scope.",
                   verdict=U, evidence={"modelled": ["mu", "fog_visibility_m", "illuminance_lux", "occlusion"]}),
              parent=si)
        # a goal whose children include an assumption is UNKNOWN unless it is
        # already VIOLATED -- which k_all over (evidence..., U) does by itself

    # S2: the domain boundary argument
    s2 = c.add(Node("S2", "Strategy",
                    "Argument over the domain boundary: what was executed is inside the ODD and "
                    "the ODD has been covered by what was executed"), parent=g1)
    outside = [sc.id for sc, v in zip(corpus, verdicts) if v["verdict"] == F]
    g21 = c.add(Node("G2.1", "Goal", "Every executed scenario lies inside the declared ODD",
                     verdict=F if outside else T,
                     evidence={"outside": len(outside), "examples": outside[:5], "of": len(corpus)}),
                parent=s2)
    if outside:
        c.add(Node("D1", "Defeater",
                   f"{len(outside)} executed scenario(s) leave the declared ODD, so the argument does not "
                   "cover them: either the ODD is understated or the corpus is out of scope",
                   verdict=F, evidence={"scenarios": outside[:10]}), parent=g21)
    from .odd import coverage as _cov

    rep = _cov(odd, corpus)
    g22 = c.add(Node("G2.2", "Goal",
                     f"The declared ODD is exercised by executed scenarios at "
                     f"{rep['marginal_coverage'] * 100:.1f}% of its axis bins "
                     f"({rep['axes']} axes, {rep['bins_per_axis']} bins each)",
                     verdict=T if rep["marginal_coverage"] >= 0.99 else U,
                     evidence={"marginal_coverage": rep["marginal_coverage"],
                               "product_coverage": rep["product_coverage"],
                               "axes": rep["axes"],
                               "cells_observed": rep["cells_observed"],
                               "cells_total": rep["cells_total"],
                               "empty_bins_by_axis": {k: len(v) for k, v in rep["empty_bins_by_axis"].items()}}),
                parent=s2)
    n_empty = sum(len(v) for v in rep["empty_bins_by_axis"].values())
    if n_empty:
        c.add(Node("D2", "Defeater",
                   f"{n_empty} axis-bin(s) of the declared ODD were never exercised; the argument "
                   "over them is UNKNOWN and is reported as such rather than assumed safe",
                   verdict=U, evidence={"uncovered_examples": rep["uncovered_cells"]}), parent=g22)
    c.resolve()
    return c


def to_sacm(case: Case) -> dict:
    """SACM-shaped export: an argument package with claims, evidence and the
    AssertedRelationship edges. Kept structural rather than schema-complete, and
    the mapping is stated so the gap is visible: SACM's ArgumentReasoning /
    AssertedInference / AssertedContext map to Strategy / solvedBy / inContextOf,
    and SACM's 'CounterClaim' maps to Defeater with the same verdict semantics."""
    d = case.to_dict()
    gid = {n["id"]: f"GID-{n['id']}" for n in d["nodes"]}
    return {
        "sacm-version": "2.3",
        "argumentPackage": d["name"],
        "artifactReference": [{"gid": gid[n["id"]], "name": n["evidence"].get("scenario", n["id"]),
                               "description": n["text"]} for n in d["nodes"] if n["kind"] == "Solution"],
        "claim": [{"gid": gid[n["id"]], "name": n["id"], "content": n["text"],
                   "verdict": n["verdict"],
                   "reasoning": [c for c in n["children"] if case.nodes[c].kind == "Strategy"]}
                  for n in d["nodes"] if n["kind"] in ("Goal", "Claim")],
        "argumentReasoning": [{"gid": gid[n["id"]], "content": n["text"],
                               "structure": n["children"]}
                              for n in d["nodes"] if n["kind"] == "Strategy"],
        "assertedInference": [{"source": gid[p], "target": gid[c]}
                              for n in d["nodes"] for c in n["children"]
                              for p in [n["id"]] if case.nodes[c].kind in ("Goal", "Solution", "Strategy")],
        "assertedContext": [{"source": gid[n["id"]], "target": "GID-C1"}
                            for n in d["nodes"] if n["id"] != "C1"],
        "counterClaim": [{"gid": gid[n["id"]], "content": n["text"], "verdict": n["verdict"]}
                         for n in d["nodes"] if n["kind"] == "Defeater"],
        "assumption": [{"gid": gid[n["id"]], "content": n["text"]}
                       for n in d["nodes"] if n["kind"] == "Assumption"],
    }


def find_deficient(case: Case) -> list:
    """The review list: every node that is not SAT, hardest first. This is what a
    person acts on, and it is the reason a three-valued case is worth the extra
    value over a two-valued one."""
    case.resolve()
    rank = {F: 0, U: 1, T: 2}
    out = [{"id": n.id, "kind": n.kind, "verdict": n.verdict, "text": n.text,
            "evidence": n.evidence} for n in case.nodes.values() if n.verdict != T]
    return sorted(out, key=lambda d: (rank[d["verdict"]], d["kind"]))
