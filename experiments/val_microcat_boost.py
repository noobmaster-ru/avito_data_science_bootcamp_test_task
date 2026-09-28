"""Проверка: вероятность микрокатегории как множитель к скору первой стадии (вредит; см. README)."""
import pickle, numpy as np
from cg.config import ART
from cg.sources.memory import ClickMemory
from cg.boost import Booster, location_centroids
from cg.metrics import report
from cg.fusion import fuse
d = pickle.load(open(ART / "val_split.pkl", "rb")); A, V, corpus, rel, seen_q = d["A"], d["V"], d["corpus"], d["rel"], d["seen_q"]
c = pickle.load(open(ART / "val_cands_v1.pkl", "rb")); mc = pickle.load(open(ART / "val_mc_proba.pkl", "rb"))
cidx = {i: n for n, i in enumerate(corpus.item_id)}
mem = ClickMemory(A, cidx); cent = location_centroids(corpus, A)
for a in (0, 5, 10, 20):
    bo = Booster(corpus, mem.item_pop, cent, mc["classes"], a=a, b=0, c=0, d=0); bo.prepare(V, mc["P"])
    f = fuse({"bm25": c["bm25"].tolist()}, {"bm25": 1.0}, forced=c["mem"], quota=5, filler=c["prior"], boost=bo)
    print(f"microcat boost a={a}:", report(f, rel, seen_q))
