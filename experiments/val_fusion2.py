"""Проверка весов слияния двух энкодеров и полноты пула (Recall@300/500) на подвыборке."""
import time, pickle, numpy as np
from cg.config import ART, SEED
from cg.pipeline import bm25_item_text, dense_query_text
from cg.sources.memory import ClickMemory
from cg.sources.bm25 import BM25Source
from cg.boost import Booster, location_centroids
from cg.scoring import MultiScorer
from cg.metrics import report
t0 = time.time()
d = pickle.load(open(ART/"val_split.pkl", "rb")); A, V, corpus, rel, seen_q = d["A"], d["V"], d["corpus"], d["rel"], d["seen_q"]
cidx = {i: n for n, i in enumerate(corpus.item_id)}
mem = ClickMemory(A, cidx); cent = location_centroids(corpus, A)
slug_ft = str(ART / "models/e5s_ft_v1").replace("/", "_")
q1 = np.load(ART/"emb/intfloat_multilingual-e5-small_valq_v1.npy"); c1 = np.load(ART/"emb/intfloat_multilingual-e5-small_valcorpus_v1.npy")
q2 = np.load(ART/f"emb/{slug_ft}_valq_ft1.npy"); c2 = np.load(ART/f"emb/{slug_ft}_valcorpus_ft1.npy")
bm = BM25Source(bm25_item_text(corpus).tolist())
sub = np.sort(np.random.default_rng(SEED).choice(len(V), 2500, replace=False))
qt = [dense_query_text(V).tolist()[i] for i in sub]; rel_s = [rel[i] for i in sub]; seen_s = seen_q[sub]
bo = Booster(corpus, mem.item_pop, cent, b=5, c=0.5, d=5, scale_km=100); bo.prepare(V.iloc[sub].reset_index(drop=True))
for name, dense in [("base w1=1 t50 | w2=1 t10", [(c1, 50, 1.0), (c2, 10, 1.0)]), ("w2=1.5", [(c1, 50, 1.0), (c2, 10, 1.5)]),
                    ("w2=2", [(c1, 50, 1.0), (c2, 10, 2.0)]), ("t1=20", [(c1, 20, 1.0), (c2, 10, 1.0)]), ("w1=0.5", [(c1, 50, 0.5), (c2, 10, 1.0)])]:
    r = MultiScorer(bm, dense, bo).run(qt, [q1[sub], q2[sub]], K=500, boost_all=bo.all)
    print(name, report(r, rel_s, seen_s, ks=(50, 300, 500)), round(time.time()-t0), flush=True)
