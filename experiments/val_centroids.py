"""Проверка: центроиды локаций поиска из train (search_location_id → медиана координат кликнутых объявлений) для локаций без объявлений в корпусе."""
import time, pickle, numpy as np, pandas as pd
from cg.config import ART, SEED
from cg.pipeline import bm25_item_text, dense_query_text
from cg.sources.memory import ClickMemory
from cg.sources.bm25 import BM25Source
from cg.boost import Booster, location_centroids, _median_xy
from cg.scoring import FullScorer
from cg.metrics import report
t0 = time.time()
d = pickle.load(open(ART/"val_split.pkl", "rb")); A, V, corpus, rel, seen_q = d["A"], d["V"], d["corpus"], d["rel"], d["seen_q"]
cidx = {i: n for n, i in enumerate(corpus.item_id)}
mem = ClickMemory(A, cidx)
cent = _median_xy(pd.concat([corpus, A[corpus.columns.intersection(A.columns)]], ignore_index=True), "item_location_id")  # только по item_location_id
s = A.assign(lat=pd.to_numeric(A.item_latitude, errors="coerce"), lon=pd.to_numeric(A.item_longitude, errors="coerce")).dropna(subset=["lat", "lon"])
cent_search = s.groupby(s.search_location_id.fillna(-1).astype(int))[["lat", "lon"]].median()
cent2 = pd.concat([cent, cent_search[~cent_search.index.isin(cent.index)]])
q_loc = V.search_location_id.fillna(-1).astype(int)
print("coverage corpus-only", round(q_loc.isin(cent.index).mean(), 3), "with train search locs", round(q_loc.isin(cent2.index).mean(), 3))
q_emb = np.load(ART/"emb/intfloat_multilingual-e5-small_valq_v1.npy"); c_emb = np.load(ART/"emb/intfloat_multilingual-e5-small_valcorpus_v1.npy")
bm = BM25Source(bm25_item_text(corpus).tolist())
sub = np.sort(np.random.default_rng(SEED).choice(len(V), 2500, replace=False))
qt = [dense_query_text(V).tolist()[i] for i in sub]; rel_s = [rel[i] for i in sub]; seen_s = seen_q[sub]
for name, cc in [("corpus-only", cent), ("corpus+train-search", cent2), ("train-search-first", pd.concat([cent_search, cent[~cent.index.isin(cent_search.index)]]))]:
    bo = Booster(corpus, mem.item_pop, cc, b=5, c=0.5, d=5, scale_km=100); bo.prepare(V.iloc[sub].reset_index(drop=True))
    r = FullScorer(bm, c_emb, bo).run(qt, q_emb[sub], K=50, tau=50, boost_all=bo.all)
    print(name, report(r, rel_s, seen_s), round(time.time()-t0), flush=True)
