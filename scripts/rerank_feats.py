"""Пул для реранкера: топ-300 полного скоринга на V с компонентами скоров.
Аргументы: имя пула, затем четвёрки <ключ эмбеддингов> <slug модели> <tau> <вес> для каждого bi-encoder'а; POOL_K — размер пула."""
import time, pickle, sys, numpy as np, pandas as pd
from cg.config import ART
from cg.pipeline import bm25_item_text, dense_query_text
from cg.sources.memory import ClickMemory
from cg.sources.bm25 import BM25Source
from cg.boost import Booster, location_centroids
from cg.scoring import MultiScorer
t0 = time.time()
out = sys.argv[1]
import os
specs = [tuple(sys.argv[i:i + 4]) for i in range(2, len(sys.argv), 4)]
pool_k = int(os.environ.get("POOL_K", 300))
d = pickle.load(open(ART/"val_split.pkl", "rb")); A, V, corpus, rel, seen_q = d["A"], d["V"], d["corpus"], d["rel"], d["seen_q"]
cidx = {i: n for n, i in enumerate(corpus.item_id)}
mem = ClickMemory(A, cidx); cent = location_centroids(corpus, A)
q_embs = [np.load(ART/f"emb/{slug}_valq_{key}.npy") for key, slug, tau, w in specs]
dense = [(np.load(ART/f"emb/{slug}_valcorpus_{key}.npy"), float(tau), float(w)) for key, slug, tau, w in specs]
bm = BM25Source(bm25_item_text(corpus).tolist())
bo = Booster(corpus, mem.item_pop, cent, b=5, c=0.5, d=5, scale_km=100); bo.prepare(V)
fs = MultiScorer(bm, dense, bo)
top, feats = fs.run(dense_query_text(V).tolist(), q_embs, K=pool_k, boost_all=bo.all, details=True)
print("scored", round(time.time()-t0), flush=True)
pickle.dump(dict(top=top, feats=feats), open(ART/f"{out}.pkl", "wb"))
