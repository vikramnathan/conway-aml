"""Step 3: guilty-group clustering + group-matching evaluation.

A *guilty group* is the transitive closure of guilty events under a
"guilty-counterparty-within-T" relation:

  - A guilty event is (node n, time t) that also carries the counterparty m of
    its transaction. "Guilty" = predicted (model score >= threshold) or
    ground-truth (label == 1); the two are built independently.
  - Two guilty events (n, t) and (m, t') are directly linked iff m is the
    counterparty of n's guilty transaction (or vice-versa) AND |t - t'| < T.
  - A group is a connected component under that relation. Chains extend groups:
    (n,t)-(m,t')-(p,t'') can put n and p in one group even if |t - t''| > T.

Because vertices are individual guilty *events*, a node whose guilty events are
far apart in time (and don't chain) can appear in more than one group — but at
any single point in time it belongs to exactly one group.

Group node-set = the guilty nodes in the component; group time-span = [min t,
max t] over its events.

Matching + penalty. A predicted group P matches a GT group G iff their node sets
intersect AND their time spans overlap. Best match maximises node Jaccard:

    penalty(P) = 1 - Jaccard(nodes_P, nodes_G)   if a match exists
    penalty(P) = 1                                if no match

Headline = mean penalty over predicted groups (precision side); recall side is
the symmetric mean over GT groups so misses are visible.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime


def _parse(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")


@dataclass
class Group:
    nodes: frozenset       # guilty nodes in this component
    start: datetime        # min event time
    end: datetime          # max event time
    size: int              # number of guilty events


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        r = x
        while self.p[r] != r:
            r = self.p[r]
        while self.p[x] != r:      # path compression
            self.p[x], x = r, self.p[x]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def build_groups(guilty_events: list[tuple], T_seconds: float) -> list[Group]:
    """guilty_events: list of (node, datetime, counterparty). Returns closure groups.

    Links event i to every guilty event of node `counterparty_i` within T seconds,
    then takes connected components.
    """
    n = len(guilty_events)
    uf = _UF(n)

    # index events by their node so we can find a counterparty's guilty events
    by_node: dict = defaultdict(list)
    for i, (node, t, _cp) in enumerate(guilty_events):
        by_node[node].append(i)

    for i, (node, t, cp) in enumerate(guilty_events):
        if cp is None:
            continue
        for j in by_node.get(cp, ()):
            if i == j:
                continue
            _n2, t2, _cp2 = guilty_events[j]
            if abs((t - t2).total_seconds()) < T_seconds:
                uf.union(i, j)

    comps: dict = defaultdict(list)
    for i in range(n):
        comps[uf.find(i)].append(i)

    groups = []
    for members in comps.values():
        nodes = set()
        tmin = tmax = None
        for i in members:
            node, t, _cp = guilty_events[i]
            nodes.add(node)
            tmin = t if tmin is None or t < tmin else tmin
            tmax = t if tmax is None or t > tmax else tmax
        groups.append(Group(frozenset(nodes), tmin, tmax, len(members)))
    return groups


def jaccard(x: frozenset, y: frozenset) -> float:
    if not x and not y:
        return 1.0
    u = len(x | y)
    return len(x & y) / u if u else 0.0


def _time_overlap(a: Group, b: Group) -> bool:
    return a.start <= b.end and b.start <= a.end


def match_penalty(pred: list[Group], gt: list[Group]) -> dict:
    """Match predicted vs GT groups; penalty = 1 - Jaccard(nodes) if matched else 1.

    Candidate matches are found via a node -> groups inverted index (necessary
    condition: shared node), then verified with an exact time-span overlap check.
    After aggregation each node maps to few groups, so the index does not blow up.
    """
    def side(query: list[Group], targets: list[Group]) -> list[float]:
        index: dict = defaultdict(list)
        for ti, g in enumerate(targets):
            for node in g.nodes:
                index[node].append(ti)
        pens = []
        for q in query:
            cand = set()
            for node in q.nodes:
                cand.update(index.get(node, ()))
            best_j = 0.0
            matched = False
            for ti in cand:
                g = targets[ti]
                if _time_overlap(q, g):   # node overlap guaranteed by index
                    matched = True
                    j = jaccard(q.nodes, g.nodes)
                    if j > best_j:
                        best_j = j
            pens.append(1.0 - best_j if matched else 1.0)
        return pens

    prec = side(pred, gt)      # per predicted group (headline)
    rec = side(gt, pred)       # per GT group (misses -> 1)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    return {
        "precision_penalty": mean(prec),
        "recall_penalty": mean(rec),
        "combined_penalty": mean(prec + rec),
        "n_pred_groups": len(pred),
        "n_gt_groups": len(gt),
        "n_pred_matched": sum(1 for p in prec if p < 1.0),
        "n_gt_matched": sum(1 for r in rec if r < 1.0),
    }


def guilty_events_from_rows(rows: list[dict], time_key="timestamp",
                            node_key="account", cp_key="counterparty_account") -> list[tuple]:
    """Turn flagged rows into (node, datetime, counterparty) tuples."""
    return [(r[node_key], _parse(r[time_key]), r.get(cp_key)) for r in rows]
