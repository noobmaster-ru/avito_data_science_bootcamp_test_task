import time, pickle, itertools, numpy as np, pandas as pd
from cg.config import ART
from cg.sources.memory import ClickMemory
from cg.boost import Booster, location_centroids
from cg.metrics import report
from cg.fusion import fuse
t0 = time.time()
d = pickle.load(open(ART/"val_split.pkl", "rb")); A, V, corpus, rel, seen_q = d["A"], d["V"], d["corpus"], d["rel"], d["seen_q"]
c = pickle.load(open(ART/"val_cands_v1.pkl", "rb"))
cidx = {i: n for n, i in enumerate(corpus.item_id)}
mem = ClickMemory(A, cidx)
cent = location_centroids(corpus, A)
src = {"bm25": c["bm25"].tolist()}
best = (0, None)
for b, cc, dd, s in [(2,0.5,0,30),(3,0.5,0,30),(5,0.5,0,30),(8,0.5,0,30),(5,1.0,0,30),(5,0,0,30),(5,0.5,2,30),(5,0.5,5,30),(5,0.5,5,10),(5,0.5,5,100),(3,0.5,5,30),(8,0.5,5,30),(8,1.0,8,30)]:
    bo = Booster(corpus, mem.item_pop, cent, b=b, c=cc, d=dd, scale_km=s); bo.prepare(V)
    f = fuse(src, {"bm25": 1.0}, forced=c["mem"], quota=20, filler=c["prior"], boost=bo)
    r = report(f, rel, seen_q); print(f"b={b} c={cc} d={dd} s={s}:", r, flush=True)
    if r["r@50"] > best[0]: best = (r["r@50"], (b, cc, dd, s))
print("best", best, "t", round(time.time()-t0))
