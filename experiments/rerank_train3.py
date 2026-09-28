"""Более простые конфигурации реранкера на закэшированных признаках (2 фолда)."""
import time, pickle, numpy as np, pandas as pd, lightgbm as lgb
from cg.config import ART
from cg.rerank import FEATS
from cg.metrics import report
t0 = time.time()
d = pickle.load(open(ART/"val_split.pkl", "rb")); V, rel, seen_q = d["V"], d["rel"], d["seen_q"]
top = pickle.load(open(ART/"val_rerank_pool.pkl", "rb"))["top"]
X = pd.read_parquet(ART/"val_rerank_X.parquet"); y = np.load(ART/"val_rerank_y.npy")
F = [f for f in FEATS if f in X.columns]
qid = np.concatenate([[qi] * len(items) for qi, items in enumerate(top)]); fold = qid % 2
def rr(p, tq):
    out, pos = [], 0
    for qi in tq:
        n = len(top[qi]); o = np.lexsort((np.arange(n), -p[pos:pos+n])); out.append([top[qi][i] for i in o[:50]]); pos += n
    return out
base = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=6)
for name, params in [("300/0.05/31", dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100)),
                     ("600/0.05/31", dict(n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=100)),
                     ("400/0.05/63/mc200", dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=200)),
                     ("800/0.03/63", dict(n_estimators=800, learning_rate=0.03, num_leaves=63, min_child_samples=50))]:
    res = []
    for f in (0, 1):
        m = lgb.LGBMClassifier(**params, **base).fit(X[fold != f][F], y[fold != f])
        tq = [qi for qi in range(len(V)) if qi % 2 == f]
        res.append(report(rr(m.predict_proba(X[fold == f][F])[:, 1], tq), [rel[qi] for qi in tq], seen_q[tq])["r@50"])
    print(f"{name}: folds {res} mean {np.mean(res):.4f}", round(time.time()-t0), flush=True)
