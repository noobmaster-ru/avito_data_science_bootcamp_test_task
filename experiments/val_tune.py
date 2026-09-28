import time, pickle, numpy as np
from cg.config import ART, SEED
from cg.pipeline import bm25_item_text, dense_query_text
from cg.sources.memory import ClickMemory
from cg.sources.bm25 import BM25Source
from cg.boost import Booster, location_centroids
from cg.scoring import FullScorer
from cg.metrics import report
t0 = time.time()
d = pickle.load(open(ART/"val_split.pkl", "rb")); A, V, corpus, rel, seen_q = d["A"], d["V"], d["corpus"], d["rel"], d["seen_q"]
cidx = {i: n for n, i in enumerate(corpus.item_id)}
mem = ClickMemory(A, cidx); cent = location_centroids(corpus, A)
q_emb = np.load(ART/"emb/intfloat_multilingual-e5-small_valq_v1.npy"); c_emb = np.load(ART/"emb/intfloat_multilingual-e5-small_valcorpus_v1.npy")
bm = BM25Source(bm25_item_text(corpus).tolist())
sub = np.sort(np.random.default_rng(SEED).choice(len(V), 2500, replace=False))
qt = [dense_query_text(V).tolist()[i] for i in sub]; qe = q_emb[sub]; rel_s = [rel[i] for i in sub]; seen_s = seen_q[sub]
def ev(name, bo, K=50, **kw):
    fs = FullScorer(bm, c_emb, bo)
    r = fs.run(qt, qe, K=K, boost_all=bo.all, **kw)
    print(name, report(r, rel_s, seen_s, ks=(50, 100, 300) if K >= 300 else (50,)), round(time.time()-t0), flush=True)
def B(**kw):
    bo = Booster(corpus, mem.item_pop, cent, **kw); bo.prepare(V.iloc[sub].reset_index(drop=True)); return bo
base = dict(b=5, c=0.5, d=5, scale_km=100)
ev("tau50 1:1 K300 (ceiling)", B(**base), K=300, tau=50)
ev("tau100 1:1", B(**base), tau=100)
ev("tau200 1:1", B(**base), tau=200)
ev("tau100 1:2", B(**base), tau=100, w_dense=2.0)
ev("tau100 2:1", B(**base), tau=100, w_bm25=2.0)
ev("tau100 b8 d8", B(b=8, c=0.5, d=8, scale_km=100), tau=100)
ev("tau100 b3 d3", B(b=3, c=0.5, d=3, scale_km=100), tau=100)
ev("tau100 c1", B(b=5, c=1.0, d=5, scale_km=100), tau=100)
ev("tau100 c0.25", B(b=5, c=0.25, d=5, scale_km=100), tau=100)
ev("tau100 s50", B(b=5, c=0.5, d=5, scale_km=50), tau=100)
ev("tau100 s200", B(b=5, c=0.5, d=5, scale_km=200), tau=100)
