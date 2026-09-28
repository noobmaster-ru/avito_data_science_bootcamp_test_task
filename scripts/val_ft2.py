"""Проверка: два dense-скора (исходный e5 и дообученный) вместе с BM25 в полном скоринге."""
import time, pickle, numpy as np, bm25s
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
slug_ft = str(ART / "models/e5s_ft_v1").replace("/", "_")
q1 = np.load(ART/"emb/intfloat_multilingual-e5-small_valq_v1.npy"); c1 = np.load(ART/"emb/intfloat_multilingual-e5-small_valcorpus_v1.npy")
q2 = np.load(ART/f"emb/{slug_ft}_valq_ft1.npy"); c2 = np.load(ART/f"emb/{slug_ft}_valcorpus_ft1.npy")
bm = BM25Source(bm25_item_text(corpus).tolist())
sub = np.sort(np.random.default_rng(SEED).choice(len(V), 2500, replace=False))
qt = [dense_query_text(V).tolist()[i] for i in sub]; rel_s = [rel[i] for i in sub]; seen_s = seen_q[sub]
bo = Booster(corpus, mem.item_pop, cent, b=5, c=0.5, d=5, scale_km=100); bo.prepare(V.iloc[sub].reset_index(drop=True))
C1 = np.ascontiguousarray(c1.T.astype(np.float32)); C2 = np.ascontiguousarray(c2.T.astype(np.float32))
fs = FullScorer(bm, None, bo)
toks = bm25s.tokenize([bm._norm(q) for q in qt], stopwords=None, stemmer=bm.stemmer, show_progress=False, return_ids=False)
def run(w1, tau1, w2, tau2, K=50):
    out = []
    for qi in range(len(qt)):
        b = fs._bm25_scores(toks[qi]).astype(np.float32); s = b / max(b.max(), 1e-6)
        if w1:
            cos = q1[sub][qi].astype(np.float32) @ C1; s = s + w1 * np.exp(tau1 * (cos - cos.max()))
        if w2:
            cos = q2[sub][qi].astype(np.float32) @ C2; s = s + w2 * np.exp(tau2 * (cos - cos.max()))
        s = s * bo.all(qi); idx = np.argpartition(-s, K)[:K]; out.append(idx[np.lexsort((idx, -s[idx]))].tolist())
    return out
for name, args in [("v1 tau50 (base)", (1, 50, 0, 20)), ("ft tau10", (0, 50, 1, 10)), ("ft tau5", (0, 50, 1, 5)),
                   ("v1 tau50 + ft tau20", (1, 50, 1, 20)), ("v1 tau50 + ft tau10", (1, 50, 1, 10)), ("v1 tau50 + 0.5 ft tau10", (1, 50, 0.5, 10))]:
    print(name, report(run(*args), rel_s, seen_s), round(time.time()-t0), flush=True)
