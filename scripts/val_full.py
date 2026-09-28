import time, pickle, sys, numpy as np, pandas as pd
from cg.config import ART
from cg.pipeline import bm25_item_text, dense_query_text
from cg.sources.memory import ClickMemory
from cg.sources.bm25 import BM25Source
from cg.sources.dense import topk
from cg.boost import Booster, location_centroids
from cg.scoring import FullScorer
from cg.metrics import report
t0 = time.time()
d = pickle.load(open(ART/"val_split.pkl", "rb")); A, V, corpus, rel, seen_q = d["A"], d["V"], d["corpus"], d["rel"], d["seen_q"]
c = pickle.load(open(ART/"val_cands_v1.pkl", "rb"))
cidx = {i: n for n, i in enumerate(corpus.item_id)}
mem = ClickMemory(A, cidx); cent = location_centroids(corpus, A)
q_emb = np.load(ART/"emb/intfloat_multilingual-e5-small_valq_v1.npy"); c_emb = np.load(ART/"emb/intfloat_multilingual-e5-small_valcorpus_v1.npy")
di, ds = topk(q_emb, c_emb, 1000)
print("dense alone:", report(di.tolist(), rel, seen_q, ks=(50,300,1000)), round(time.time()-t0), flush=True)
pickle.dump(dict(dense_1000=di, dense_1000_sc=ds), open(ART/"val_cands_dense.pkl", "wb"))
bm = BM25Source(bm25_item_text(corpus).tolist())
bo = Booster(corpus, mem.item_pop, cent, b=5, c=0.5, d=5, scale_km=100); bo.prepare(V)
fs = FullScorer(bm, c_emb, bo)
qt = dense_query_text(V).tolist()
n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
sub = slice(0, n); rel_s, seen_s = rel[:n], seen_q[:n]
def ev(name, **kw):
    r = fs.run(qt[sub], q_emb[sub], K=50, boost_all=bo.all, **kw)
    print(name, report(r, rel_s, seen_s), round(time.time()-t0), flush=True)
ev("bm25 full+boost", w_dense=0)
ev("dense full+boost tau20", w_bm25=0, tau=20)
ev("bm25+dense score tau20 1:1", tau=20)
ev("bm25+dense score tau20 1:0.5", w_dense=0.5, tau=20)
ev("bm25+dense score tau50 1:1", tau=50)
ev("bm25+dense score tau10 1:1", tau=10)
ev("bm25+dense rrf 1:1", mode="rrf")
