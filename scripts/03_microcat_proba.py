"""Классификатор «текст запроса → микрокатегория» на A; вероятности для V сохраняются как признак реранкера."""
import pickle, time, numpy as np
from cg.config import ART
from cg.sources.microcat import MicrocatClassifier
t0 = time.time()
d = pickle.load(open(ART / "val_split.pkl", "rb")); A, V, seen_q = d["A"], d["V"], d["seen_q"]
mc = MicrocatClassifier().fit(A)
P = mc.predict_proba(V.q.tolist())
col = {int(m): j for j, m in enumerate(mc.classes)}
true = np.array([col.get(m, -1) for m in V.item_microcat_id.astype(int).values])
order = np.argsort(-P, axis=1)
top1, top3 = order[:, 0] == true, (order[:, :3] == true[:, None]).any(1)
print(f"microcat top-1 {top1.mean():.3f} (seen {top1[seen_q].mean():.3f}, unseen {top1[~seen_q].mean():.3f}), top-3 {top3.mean():.3f}")
pickle.dump(dict(P=P, classes=mc.classes), open(ART / "val_mc_proba.pkl", "wb"))
print("saved val_mc_proba.pkl", round(time.time() - t0), "s")
