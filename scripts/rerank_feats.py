"""Пул для реранкера: топ-300 полного скоринга на V с компонентами скоров. Аргументы: ключ эмбеддингов, имя пула."""
import time, pickle, sys, numpy as np, pandas as pd
from cg.config import ART
from cg.pipeline import bm25_item_text, dense_query_text
from cg.sources.memory import ClickMemory
from cg.sources.bm25 import BM25Source
from cg.boost import Booster, location_centroids
from cg.scoring import FullScorer
t0 = time.time()
key, out = (sys.argv + ["v1", "val_rerank_pool"])[1:3]
slug = sys.argv[3] if len(sys.argv) > 3 else "intfloat_multilingual-e5-small"
d = pickle.load(open(ART/"val_split.pkl", "rb")); A, V, corpus, rel, seen_q = d["A"], d["V"], d["corpus"], d["rel"], d["seen_q"]
cidx = {i: n for n, i in enumerate(corpus.item_id)}
mem = ClickMemory(A, cidx); cent = location_centroids(corpus, A)
q_emb = np.load(ART/f"emb/{slug}_valq_{key}.npy"); c_emb = np.load(ART/f"emb/{slug}_valcorpus_{key}.npy")
bm = BM25Source(bm25_item_text(corpus).tolist())
bo = Booster(corpus, mem.item_pop, cent, b=5, c=0.5, d=5, scale_km=100); bo.prepare(V)
fs = FullScorer(bm, c_emb, bo)
top, feats = fs.run(dense_query_text(V).tolist(), q_emb, K=300, tau=50, boost_all=bo.all, details=True)
print("scored", round(time.time()-t0), flush=True)
pickle.dump(dict(top=top, feats=feats), open(ART/f"{out}.pkl", "wb"))
