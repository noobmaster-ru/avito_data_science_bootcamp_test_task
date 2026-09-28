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
bm_idx, bm_sc = bm.retrieve(dense_query_text(V).tolist(), 1000)
print("bm25 k=1000:", report(bm_idx.tolist(), rel, seen_q, ks=(50,300,1000)), round(time.time()-t0), flush=True)
pickle.dump(dict(bm25_1000=bm_idx, bm25_1000_sc=bm_sc), open(ART/"val_cands_bm1000.pkl", "wb"))
bo = Booster(corpus, mem.item_pop, cent, b=5, c=0.5, d=5, scale_km=100); bo.prepare(V)
for k in [300, 500, 1000]:
    for quota in [0, 5, 20]:
        f = fuse({"bm25": bm_idx[:, :k].tolist()}, {"bm25": 1.0}, forced=c["mem"], quota=quota, filler=c["prior"], boost=bo)
        print(f"k={k} quota={quota}:", report(f, rel, seen_q), flush=True)
print("t", round(time.time()-t0))
