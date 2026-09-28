"""Проверка: символьный TF-IDF (char_wb 3–5 по заголовку и началу параметров) как третий скор первой стадии."""
import time, pickle, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from cg.config import ART, SEED
from cg.pipeline import bm25_item_text, dense_query_text
from cg.sources.memory import ClickMemory
from cg.sources.bm25 import BM25Source
from cg.boost import Booster, location_centroids
from cg.scoring import FullScorer
from cg.metrics import report
from cg.text import norm
t0 = time.time()
d = pickle.load(open(ART/"val_split.pkl", "rb")); A, V, corpus, rel, seen_q = d["A"], d["V"], d["corpus"], d["rel"], d["seen_q"]
cidx = {i: n for n, i in enumerate(corpus.item_id)}
mem = ClickMemory(A, cidx); cent = location_centroids(corpus, A)
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=400_000, sublinear_tf=True, dtype=np.float32)
M = vec.fit_transform([norm(t) for t in (corpus.item_title_raw + " " + corpus.item_infm_params_text.str.slice(0, 200))]).T.tocsr()
print("char tfidf", M.shape, round(time.time()-t0), flush=True)
q_emb = np.load(ART/"emb/intfloat_multilingual-e5-small_valq_v1.npy"); c_emb = np.load(ART/"emb/intfloat_multilingual-e5-small_valcorpus_v1.npy")
bm = BM25Source(bm25_item_text(corpus).tolist())
sub = np.sort(np.random.default_rng(SEED).choice(len(V), 2500, replace=False))
qt = [dense_query_text(V).tolist()[i] for i in sub]; rel_s = [rel[i] for i in sub]; seen_s = seen_q[sub]
Q = vec.transform([norm(q) for q in qt])
bo = Booster(corpus, mem.item_pop, cent, b=5, c=0.5, d=5, scale_km=100); bo.prepare(V.iloc[sub].reset_index(drop=True))
class Scorer3(FullScorer):
    """FullScorer + w_char * char_tfidf / max."""
    def run3(self, K, w_char, tau=50):
        toks = __import__("bm25s").tokenize([self.bm25._norm(q) for q in qt], stopwords=None, stemmer=self.bm25.stemmer, show_progress=False, return_ids=False)
        out = []
        for qi in range(len(qt)):
            b = self._bm25_scores(toks[qi]).astype(np.float32); b = b / max(b.max(), 1e-6)
            cos = q_emb[sub][qi].astype(np.float32) @ self.c; ds = np.exp(tau * (cos - cos.max()))
            ch = (Q[qi] @ M).toarray().ravel(); ch = ch / max(ch.max(), 1e-6)
            s = (b + ds + w_char * ch) * bo.all(qi)
            idx = np.argpartition(-s, K)[:K]; out.append(idx[np.lexsort((idx, -s[idx]))].tolist())
        return out
fs = Scorer3(bm, c_emb, bo)
for w in [0.0, 0.5, 1.0]:
    print(f"w_char={w}:", report(fs.run3(50, w), rel_s, seen_s), round(time.time()-t0), flush=True)
