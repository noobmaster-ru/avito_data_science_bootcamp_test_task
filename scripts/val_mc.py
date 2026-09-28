import time, pickle, numpy as np, pandas as pd
from cg.config import ART
from cg.sources.microcat import MicrocatClassifier
from cg.sources.memory import ClickMemory
from cg.boost import Booster
from cg.metrics import report
from cg.fusion import fuse
t0 = time.time()
d = pickle.load(open(ART/"val_split.pkl", "rb")); A, V, corpus, rel, seen_q = d["A"], d["V"], d["corpus"], d["rel"], d["seen_q"]
c = pickle.load(open(ART/"val_cands_v1.pkl", "rb"))
mc = MicrocatClassifier().fit(A)
print("mc fit", round(time.time()-t0), flush=True)
P = mc.predict_proba(V.q.tolist())
true = V.item_microcat_id.astype(int).values
col = {int(m): j for j, m in enumerate(mc.classes)}
tc = np.array([col.get(m, -1) for m in true])
order = np.argsort(-P, axis=1)
top1 = order[:, 0] == tc; top3 = (order[:, :3] == tc[:, None]).any(1); top5 = (order[:, :5] == tc[:, None]).any(1)
print(f"microcat acc top1 {top1.mean():.3f} top3 {top3.mean():.3f} top5 {top5.mean():.3f} seen top1 {top1[seen_q].mean():.3f} unseen top1 {top1[~seen_q].mean():.3f}")
pickle.dump(dict(P=P, classes=mc.classes), open(ART/"val_mc_proba.pkl", "wb"))
cidx = {i: n for n, i in enumerate(corpus.item_id)}
mem = ClickMemory(A, cidx)
src = {"bm25": c["bm25"].tolist()}
for a, b, cc in [(0,0,0), (5,0,0), (10,0,0), (20,0,0), (10,1,0), (10,2,0), (10,1,0.5), (10,1,1.0), (20,2,1.0)]:
    bo = Booster(corpus, mem.item_pop, mc.classes, a=a, b=b, c=cc); bo.prepare(V, P)
    f = fuse(src, {"bm25": 1.0}, forced=c["mem"], quota=20, filler=c["prior"], boost=bo)
    print(f"a={a} b={b} c={cc}:", report(f, rel, seen_q), flush=True)
print("done", round(time.time()-t0))
