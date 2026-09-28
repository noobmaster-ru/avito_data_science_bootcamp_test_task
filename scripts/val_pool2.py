import time, pickle, numpy as np, pandas as pd
from cg.config import ART
from cg.pipeline import bm25_item_text, dense_query_text
from cg.sources.memory import ClickMemory
from cg.sources.bm25 import BM25Source
from cg.boost import Booster, location_centroids
from cg.metrics import report
from cg.fusion import fuse
t0 = time.time()
d = pickle.load(open(ART/"val_split.pkl", "rb")); A, V, corpus, rel, seen_q = d["A"], d["V"], d["corpus"], d["rel"], d["seen_q"]
c = pickle.load(open(ART/"val_cands_v1.pkl", "rb"))
cidx = {i: n for n, i in enumerate(corpus.item_id)}
mem = ClickMemory(A, cidx); cent = location_centroids(corpus, A)
bm = BM25Source(bm25_item_text(corpus).tolist())
bm_idx, bm_sc = bm.retrieve(dense_query_text(V).tolist(), 5000)
print("bm25 k=5000:", report(bm_idx.tolist(), rel, seen_q, ks=(1000,2000,3000,5000)), round(time.time()-t0), flush=True)
pickle.dump(dict(bm25_5000=bm_idx, bm25_5000_sc=bm_sc), open(ART/"val_cands_bm5000.pkl", "wb"))
bo = Booster(corpus, mem.item_pop, cent, b=5, c=0.5, d=5, scale_km=100); bo.prepare(V)
for k in [2000, 3000, 5000]:
    f = fuse({"bm25": bm_idx[:, :k].tolist()}, {"bm25": 1.0}, forced=c["mem"], quota=5, filler=c["prior"], boost=bo)
    print(f"rank-rrf k={k}:", report(f, rel, seen_q), flush=True)
# score-based: bm25 score * boost
def score_fuse(k, bo):
    out = []
    for qi in range(len(V)):
        items = bm_idx[qi, :k]; sc = bm_sc[qi, :k].astype(np.float64)
        ok = items >= 0; items, sc = items[ok], sc[ok]
        vals = sc * bo(qi, items)
        top = items[np.lexsort((items, -vals))][:50].tolist()
        seen = set(top)
        for it in c["mem"][qi][:5]:
            if it not in seen and len(top) < 50: top.append(it); seen.add(it)
        for it in c["prior"][qi]:
            if len(top) >= 50: break
            if it not in seen: top.append(it); seen.add(it)
        out.append(top)
    return out
for k in [1000, 2000, 5000]:
    print(f"score-based k={k}:", report(score_fuse(k, bo), rel, seen_q), flush=True)
for b, cc, dd in [(3,0.5,3), (8,0.5,8), (10,1.0,10)]:
    bo2 = Booster(corpus, mem.item_pop, cent, b=b, c=cc, d=dd, scale_km=100); bo2.prepare(V)
    print(f"score-based k=5000 b={b} c={cc} d={dd}:", report(score_fuse(5000, bo2), rel, seen_q), flush=True)
print("t", round(time.time()-t0))
