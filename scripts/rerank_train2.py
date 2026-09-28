"""Сравнение вариантов реранкера на 2 фолдах V: классификатор с разными гиперпараметрами и lambdarank."""
import time, pickle, numpy as np, pandas as pd, lightgbm as lgb
from cg.config import ART
from cg.sources.memory import ClickMemory
from cg.boost import Booster, location_centroids
from cg.rerank import FeatureBuilder, Reranker, FEATS
from cg.metrics import report
t0 = time.time()
d = pickle.load(open(ART/"val_split.pkl", "rb")); A, V, corpus, rel, seen_q = d["A"], d["V"], d["corpus"], d["rel"], d["seen_q"]
pool = pickle.load(open(ART/"val_rerank_pool.pkl", "rb")); top, feats = pool["top"], pool["feats"]
mc = pickle.load(open(ART/"val_mc_proba.pkl", "rb"))
cidx = {i: n for n, i in enumerate(corpus.item_id)}
mem = ClickMemory(A, cidx); cent = location_centroids(corpus, A)
bo = Booster(corpus, mem.item_pop, cent, b=5, c=0.5, d=5, scale_km=100); bo.prepare(V)
X = FeatureBuilder(corpus, bo, mc["classes"]).build(V, top, feats, seen_q, mc["P"])
y = np.concatenate([[int(i in set(r)) for i in items] for items, r in zip(top, rel)])
qid = np.concatenate([[qi] * len(items) for qi, items in enumerate(top)])
fold = qid % 2
X.to_parquet(ART/"val_rerank_X.parquet"); np.save(ART/"val_rerank_y.npy", y)

def rerank_scores(p, tq):
    out, pos = [], 0
    for qi in tq:
        n = len(top[qi]); order = np.lexsort((np.arange(n), -p[pos:pos+n])); out.append([top[qi][i] for i in order[:50]]); pos += n
    return out

def cv(name, make, ranker=False):
    res = []
    for f in (0, 1):
        tr, te = fold != f, fold == f
        m = make()
        if ranker:
            g = np.bincount(qid[tr]); g = g[g > 0]
            m.fit(X[tr][FEATS], y[tr], group=g)
            p = m.predict(X[te][FEATS])
        else:
            m.fit(X[tr][FEATS], y[tr]); p = m.predict_proba(X[te][FEATS])[:, 1]
        tq = [qi for qi in range(len(V)) if qi % 2 == f]
        res.append(report(rerank_scores(p, tq), [rel[qi] for qi in tq], seen_q[tq])["r@50"])
    print(f"{name}: folds {res} mean {np.mean(res):.4f}", round(time.time()-t0), flush=True)
    return np.mean(res)

base = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=8)
cv("clf 600/0.05/63", lambda: lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=50, **base))
cv("rank 600/0.05/63", lambda: lgb.LGBMRanker(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=50, lambdarank_truncation_level=60, **base), ranker=True)
cv("clf 1500/0.03/127", lambda: lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.03, num_leaves=127, min_child_samples=100, **base))
cv("rank 1500/0.03/127", lambda: lgb.LGBMRanker(n_estimators=1500, learning_rate=0.03, num_leaves=127, min_child_samples=100, lambdarank_truncation_level=60, **base), ranker=True)
