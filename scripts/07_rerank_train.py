"""Реранкер LightGBM: кросс-валидация по запросам V (2 фолда), затем модель на всём V для бенчмарка.
Аргументы: имя пула (по умолчанию val_rerank_pool_final), имя модели (по умолчанию reranker_final)."""
import pickle, sys, time, numpy as np
from cg.config import ART
from cg.sources.memory import ClickMemory
from cg.boost import Booster, location_centroids
from cg.rerank import FeatureBuilder, Reranker, FEATS
from cg.metrics import report
t0 = time.time()
pool_name, out = (sys.argv + ["val_rerank_pool_final", "reranker_final"])[1:3]
d = pickle.load(open(ART / "val_split.pkl", "rb")); A, V, corpus, rel, seen_q = d["A"], d["V"], d["corpus"], d["rel"], d["seen_q"]
pool = pickle.load(open(ART / f"{pool_name}.pkl", "rb")); top, feats = pool["top"], pool["feats"]
mc = pickle.load(open(ART / "val_mc_proba.pkl", "rb"))
cidx = {i: n for n, i in enumerate(corpus.item_id)}
mem = ClickMemory(A, cidx); cent = location_centroids(corpus, A)
bo = Booster(corpus, mem.item_pop, cent, b=5, c=0.5, d=5, scale_km=100); bo.prepare(V)
print("pool (first stage):", report(top, rel, seen_q, ks=(50, 100, 300, 500)), flush=True)
X = FeatureBuilder(corpus, bo, mc["classes"]).build(V, top, feats, seen_q, mc["P"], mem)
y = np.concatenate([[int(i in set(r)) for i in items] for items, r in zip(top, rel)])
qid = np.concatenate([[qi] * len(items) for qi, items in enumerate(top)])
fold, res = qid % 2, []
for f in (0, 1):
    rr = Reranker().fit(X[fold != f], y[fold != f])
    tq = [qi for qi in range(len(V)) if qi % 2 == f]
    r = report(rr.rerank(X[fold == f], [top[qi] for qi in tq]), [rel[qi] for qi in tq], seen_q[tq])
    res.append(r["r@50"])
    print(f"fold {f}: after rerank {r} | first stage {report([top[qi] for qi in tq], [rel[qi] for qi in tq], seen_q[tq])['r@50']}", flush=True)
    print("top features:", [f for f, _ in sorted(zip(FEATS, rr.model.feature_importances_), key=lambda x: -x[1])[:8]])
print(f"mean Recall@50 after rerank: {np.mean(res):.4f}")
pickle.dump(Reranker().fit(X, y), open(ART / f"{out}.pkl", "wb"))
print(f"saved {out}.pkl", round(time.time() - t0), "s")
