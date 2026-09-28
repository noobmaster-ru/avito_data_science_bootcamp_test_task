"""Оценка дообученного bi-encoder на валидации и подготовка его эмбеддингов для бенчмарка."""
import time, pickle, sys, numpy as np, shutil
from cg.config import ART, SEED
from cg.data import load
from cg.pipeline import bm25_item_text, dense_item_text, dense_query_text
from cg.sources.memory import ClickMemory
from cg.sources.bm25 import BM25Source
from cg.sources.dense import DenseSource, topk
from cg.boost import Booster, location_centroids
from cg.scoring import FullScorer
from cg.metrics import report
t0 = time.time()
model_path = sys.argv[1] if len(sys.argv) > 1 else str(ART / "models/e5s_ft_v1"); key = sys.argv[2] if len(sys.argv) > 2 else "ft1"
d = pickle.load(open(ART/"val_split.pkl", "rb")); A, V, corpus, rel, seen_q = d["A"], d["V"], d["corpus"], d["rel"], d["seen_q"]
bq, bi = load("benchmark_queries"), load("benchmark_items")
ds = DenseSource(model_path)
q_emb = ds.encode(dense_query_text(V).tolist(), "query: ", f"valq_{key}", batch=256)
bq_emb = ds.encode(dense_query_text(bq).tolist(), "query: ", f"benchq_{key}", batch=256)
c_emb = ds.encode(dense_item_text(corpus).tolist(), "passage: ", f"valcorpus_{key}", batch=256)
print("encoded", round(time.time()-t0), flush=True)
assert (corpus.item_id.values[:len(bi)] == bi.item_id.values).all()
np.save(ART / "emb" / f"{ds.slug}_benchcorpus_{key}.npy", c_emb[:len(bi)])
np.save(ART / "emb" / f"{ds.slug}_benchq_benchcorpus_{key}.npy", bq_emb)
di, _ = topk(q_emb, c_emb, 1000)
print("dense-ft alone:", report(di.tolist(), rel, seen_q, ks=(50, 300, 1000)), flush=True)
cidx = {i: n for n, i in enumerate(corpus.item_id)}
mem = ClickMemory(A, cidx); cent = location_centroids(corpus, A)
bm = BM25Source(bm25_item_text(corpus).tolist())
sub = np.sort(np.random.default_rng(SEED).choice(len(V), 2500, replace=False))
qt = [dense_query_text(V).tolist()[i] for i in sub]; rel_s = [rel[i] for i in sub]; seen_s = seen_q[sub]
bo = Booster(corpus, mem.item_pop, cent, b=5, c=0.5, d=5, scale_km=100); bo.prepare(V.iloc[sub].reset_index(drop=True))
fs = FullScorer(bm, c_emb, bo)
for tau, wd in [(50, 1.0), (20, 1.0), (50, 2.0)]:
    r = fs.run(qt, q_emb[sub], K=50, tau=tau, w_dense=wd, boost_all=bo.all)
    print(f"bm25+dense-ft tau{tau} w_dense{wd}:", report(r, rel_s, seen_s), round(time.time()-t0), flush=True)
r = fs.run(qt, q_emb[sub], K=50, w_bm25=0.0, tau=50, boost_all=bo.all)
print("dense-ft only + boost:", report(r, rel_s, seen_s), round(time.time()-t0), flush=True)
