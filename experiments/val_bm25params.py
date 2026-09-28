"""Проверка параметров BM25 (k1, b) и веса заголовка в полном скоринге."""
import time, pickle, numpy as np, pandas as pd
from cg.config import ART, SEED
from cg.pipeline import dense_query_text
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
sub = np.sort(np.random.default_rng(SEED).choice(len(V), 2500, replace=False))
qt = [dense_query_text(V).tolist()[i] for i in sub]; rel_s = [rel[i] for i in sub]; seen_s = seen_q[sub]
bo = Booster(corpus, mem.item_pop, cent, b=5, c=0.5, d=5, scale_km=100); bo.prepare(V.iloc[sub].reset_index(drop=True))
def texts(tw, plen=600, dlen=300):
    return ((corpus.item_title_raw + " . ") * tw + corpus.item_infm_params_text.str.slice(0, plen) + " . " + corpus.item_description_raw.str.slice(0, dlen)).tolist()
import sys
SWEEP = {"1": [("base t3 p600 d300 k1=1.2 b=0.75", 3, 600, 300, 1.2, 0.75), ("k1=0.9 b=0.5", 3, 600, 300, 0.9, 0.5),
               ("k1=1.5 b=0.9", 3, 600, 300, 1.5, 0.9), ("title x5", 5, 600, 300, 1.2, 0.75),
               ("title x1", 1, 600, 300, 1.2, 0.75), ("p300 d600", 3, 300, 600, 1.2, 0.75), ("p1000 d500", 3, 1000, 500, 1.2, 0.75)],
         "2": [("p300 d1000", 3, 300, 1000, 1.2, 0.75), ("p300 d2000", 3, 300, 2000, 1.2, 0.75), ("p300 d6000", 3, 300, 6000, 1.2, 0.75),
               ("p600 d1000", 3, 600, 1000, 1.2, 0.75), ("p150 d1000", 3, 150, 1000, 1.2, 0.75)]}
for name, tw, plen, dlen, k1, b in SWEEP[sys.argv[1] if len(sys.argv) > 1 else "1"]:
    bm = BM25Source(texts(tw, plen, dlen), k1=k1, b=b)
    r = FullScorer(bm, c_emb, bo).run(qt, q_emb[sub], K=50, tau=50, boost_all=bo.all)
    print(name, report(r, rel_s, seen_s), round(time.time()-t0), flush=True)
