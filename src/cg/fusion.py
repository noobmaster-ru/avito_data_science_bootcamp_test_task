import numpy as np


def fuse(sources: dict, weights: dict, forced=None, quota: int = 20, filler=None, K: int = 50, k_rrf: int = 60,
         allowed=None, boost=None) -> list:
    """Слияние: до quota объявлений из forced (память), затем взвешенный RRF по источникам с множителями boost, добор из filler."""
    n = len(forced) if forced is not None else len(next(iter(sources.values())))
    out = []
    for qi in range(n):
        ok = allowed[qi] if allowed is not None else None
        res, seen = [], set()

        def add(it):
            if it >= 0 and it not in seen and (ok is None or ok[it]):
                seen.add(it)
                res.append(it)

        if forced is not None:
            for it in forced[qi][:quota]:
                add(it)
        scores = {}
        for name, lists in sources.items():
            w = weights.get(name, 0.0)
            if w:
                for r, it in enumerate(lists[qi]):
                    if it >= 0:
                        scores[it] = scores.get(it, 0.0) + w / (k_rrf + r + 1)
        if scores:
            items = np.fromiter(scores.keys(), dtype=np.int64, count=len(scores))
            vals = np.fromiter(scores.values(), dtype=np.float64, count=len(scores))
            if boost is not None:
                vals = vals * boost(qi, items)
            for it in items[np.lexsort((items, -vals))]:
                if len(res) >= K:
                    break
                add(int(it))
        if filler is not None:
            for it in filler[qi]:
                if len(res) >= K:
                    break
                add(it)
        out.append(res[:K])
    return out
