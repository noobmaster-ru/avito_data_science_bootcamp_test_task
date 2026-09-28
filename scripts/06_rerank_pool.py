"""Пул для реранкера на V: топ-POOL_K полного скоринга (BM25 + bi-encoder'ы + бусты) с компонентами скоров.
Аргументы: имя пула, затем четвёрки <ключ эмбеддингов> <slug модели> <tau> <вес>; без аргументов — финальная конфигурация."""
import os, pickle, sys, time, numpy as np
from cg.config import ART, DENSE_MODEL, FT_MODEL
from cg.pipeline import bm25_item_text, dense_query_text
from cg.sources.memory import ClickMemory
from cg.sources.bm25 import BM25Source
from cg.boost import Booster, location_centroids
from cg.scoring import MultiScorer
t0 = time.time()
out = sys.argv[1] if len(sys.argv) > 1 else "val_rerank_pool_final"
specs = [tuple(sys.argv[i:i + 4]) for i in range(2, len(sys.argv), 4)] or [
    ("v1", DENSE_MODEL.replace("/", "_"), "50", "1.0"), ("ft1", FT_MODEL.replace("/", "_"), "10", "2.0")]
pool_k = int(os.environ.get("POOL_K", 500))
d = pickle.load(open(ART / "val_split.pkl", "rb")); A, V, corpus = d["A"], d["V"], d["corpus"]
cidx = {i: n for n, i in enumerate(corpus.item_id)}
mem = ClickMemory(A, cidx); cent = location_centroids(corpus, A)
q_embs = [np.load(ART / f"emb/{slug}_valq_{key}.npy") for key, slug, tau, w in specs]
dense = [(np.load(ART / f"emb/{slug}_valcorpus_{key}.npy"), float(tau), float(w)) for key, slug, tau, w in specs]
bm = BM25Source(bm25_item_text(corpus).tolist())
bo = Booster(corpus, mem.item_pop, cent, b=5, c=0.5, d=5, scale_km=100); bo.prepare(V)
top, feats = MultiScorer(bm, dense, bo).run(dense_query_text(V).tolist(), q_embs, K=pool_k, boost_all=bo.all, details=True)
pickle.dump(dict(top=top, feats=feats), open(ART / f"{out}.pkl", "wb"))
print(f"saved {out}.pkl: {len(top)} queries x {pool_k} candidates", round(time.time() - t0), "s")
