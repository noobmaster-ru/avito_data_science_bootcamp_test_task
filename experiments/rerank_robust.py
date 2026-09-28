"""Устойчивость реранкера к сдвигу: качество на запросах, чьё релевантное объявление не встречалось в train, с признаками популярности и без."""
import time, pickle, numpy as np, pandas as pd, lightgbm as lgb
from cg.config import ART
from cg.sources.memory import ClickMemory
from cg.boost import Booster, location_centroids
from cg.rerank import FeatureBuilder, FEATS
from cg.metrics import recall_at_k
t0 = time.time()
d = pickle.load(open(ART/"val_split.pkl", "rb")); A, V, corpus, rel, seen_q, seen_item = d["A"], d["V"], d["corpus"], d["rel"], d["seen_q"], d["seen_item"]
pool = pickle.load(open(ART/"val_rerank_pool_c2.pkl", "rb")); top, feats = pool["top"], pool["feats"]
mc = pickle.load(open(ART/"val_mc_proba.pkl", "rb"))
cidx = {i: n for n, i in enumerate(corpus.item_id)}
mem = ClickMemory(A, cidx); cent = location_centroids(corpus, A)
bo = Booster(corpus, mem.item_pop, cent, b=5, c=0.5, d=5, scale_km=100); bo.prepare(V)
X = FeatureBuilder(corpus, bo, mc["classes"]).build(V, top, feats, seen_q, mc["P"], mem)
X.to_parquet(ART/"val_rerank_X_c2.parquet")
y = np.concatenate([[int(i in set(r)) for i in items] for items, r in zip(top, rel)])
qid = np.concatenate([[qi] * len(items) for qi, items in enumerate(top)]); fold = qid % 2
print("features", X.shape, round(time.time()-t0), flush=True)
def rr(p, tq):
    out, pos = [], 0
    for qi in tq:
        n = len(top[qi]); o = np.lexsort((np.arange(n), -p[pos:pos+n])); out.append([top[qi][i] for i in o[:50]]); pos += n
    return out
def subset(preds, tq, mask):
    idx = [k for k, qi in enumerate(tq) if mask[qi]]
    return round(recall_at_k([preds[k] for k in idx], [rel[tq[k]] for k in idx]), 4)
params = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=6)
variants = {"all": FEATS, "no_pop": [f for f in FEATS if f not in ("log_pop", "seen", "mem_level", "mem_logn")],
            "no_pop_no_boost": [f for f in FEATS if f not in ("log_pop", "seen", "mem_level", "mem_logn", "boost", "score")]}
for name, F in variants.items():
    tot, si, ui = [], [], []
    for f in (0, 1):
        m = lgb.LGBMClassifier(**params).fit(X[fold != f][F], y[fold != f])
        tq = [qi for qi in range(len(V)) if qi % 2 == f]
        preds = rr(m.predict_proba(X[fold == f][F])[:, 1], tq)
        tot.append(recall_at_k(preds, [rel[qi] for qi in tq])); si.append(subset(preds, tq, seen_item)); ui.append(subset(preds, tq, ~seen_item))
    print(f"{name}: all {np.mean(tot):.4f} | item seen in train {np.mean(si):.4f} | item unseen {np.mean(ui):.4f}", round(time.time()-t0), flush=True)
base_tot = recall_at_k(top, rel); print(f"pool (no rerank): all {base_tot:.4f} | seen-item {subset(top, list(range(len(V))), seen_item):.4f} | unseen-item {subset(top, list(range(len(V))), ~seen_item):.4f}")
