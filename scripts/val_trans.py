"""Проверка буста по таблице переходов локаций: P(локация объявления | локация поиска) из train."""
import time, pickle, numpy as np, pandas as pd
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
g = A.groupby([A.search_location_id.fillna(-1).astype(int), A.item_location_id.fillna(-1).astype(int)]).size()
tot = g.groupby(level=0).sum(); trans = (g / tot).rename("p").reset_index()
trans = {k: dict(zip(v.item_location_id, v.p)) for k, v in trans.groupby("search_location_id")}
class TransBooster(Booster):
    """Booster + множитель (1 + e * P(loc_item | loc_search) / max P)."""
    def __init__(self, *a, e=3.0, **kw):
        super().__init__(*a, **kw); self.e = e; self.item_loc_s = pd.Series(self.item_loc)
    def all(self, qi):
        m = super().all(qi)
        row = trans.get(int(self.q_loc[qi]))
        if row:
            p = self.item_loc_s.map(row).fillna(0.0).values / max(row.values())
            m = m * (1 + self.e * p)
        return m
q_emb = np.load(ART/"emb/intfloat_multilingual-e5-small_valq_v1.npy"); c_emb = np.load(ART/"emb/intfloat_multilingual-e5-small_valcorpus_v1.npy")
bm = BM25Source(bm25_item_text(corpus).tolist())
sub = np.sort(np.random.default_rng(SEED).choice(len(V), 2500, replace=False))
qt = [dense_query_text(V).tolist()[i] for i in sub]; rel_s = [rel[i] for i in sub]; seen_s = seen_q[sub]
Vs = V.iloc[sub].reset_index(drop=True)
for name, mk in [("base", lambda: Booster(corpus, mem.item_pop, cent, b=5, c=0.5, d=5, scale_km=100)),
                 ("trans e=2", lambda: TransBooster(corpus, mem.item_pop, cent, b=5, c=0.5, d=5, scale_km=100, e=2)),
                 ("trans e=5", lambda: TransBooster(corpus, mem.item_pop, cent, b=5, c=0.5, d=5, scale_km=100, e=5)),
                 ("trans e=5 b=2", lambda: TransBooster(corpus, mem.item_pop, cent, b=2, c=0.5, d=5, scale_km=100, e=5))]:
    bo = mk(); bo.prepare(Vs)
    r = FullScorer(bm, c_emb, bo).run(qt, q_emb[sub], K=50, tau=50, boost_all=bo.all)
    print(name, report(r, rel_s, seen_s), round(time.time()-t0), flush=True)
