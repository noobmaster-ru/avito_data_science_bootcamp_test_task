"""End-to-end: train + benchmark → полный скоринг корпуса (BM25 + dense, бусты) → реранкер → answer.csv."""
import json, pickle, shutil, subprocess, sys, time
import numpy as np
import pandas as pd
from cg.config import ART, ROOT, K, DENSE_MODEL
from cg.data import load
from cg.validation import prepare
from cg.pipeline import bm25_item_text, dense_item_text, dense_query_text
from cg.sources.memory import ClickMemory
from cg.sources.bm25 import BM25Source
from cg.sources.microcat import MicrocatClassifier
from cg.boost import Booster, location_centroids
from cg.scoring import FullScorer
from cg.rerank import FeatureBuilder
from cg.submit import build_answer, save

CONFIGS = {
    "v3": dict(name="v3_full_bm25_dense_tau50", model=DENSE_MODEL, emb_key="benchcorpus_v1", w_bm25=1.0, w_dense=1.0, tau=50.0,
               boost=dict(b=5, c=0.5, d=5, scale_km=100), rerank=None, pool=50),
    "v4": dict(name="v4b_rerank_centroids", model=DENSE_MODEL, emb_key="benchcorpus_v1", w_bm25=1.0, w_dense=1.0, tau=50.0,
               boost=dict(b=5, c=0.5, d=5, scale_km=100), rerank="reranker_v3.pkl", pool=300),
    "v5": dict(name="v5_rerank_ft", model=str(ART / "models/e5s_ft_v1"), emb_key="benchcorpus_ft1", w_bm25=1.0, w_dense=1.0, tau=50.0,
               boost=dict(b=5, c=0.5, d=5, scale_km=100), rerank="reranker_ft1.pkl", pool=300),
}


def embeddings(cfg, corpus, bq):
    """Эмбеддинги корпуса и запросов бенчмарка из кэша artifacts/emb; если кэша нет, считает их в подпроцессе."""
    slug = cfg["model"].replace("/", "_")
    paths = [ART / "emb" / f"{slug}_{cfg['emb_key']}.npy", ART / "emb" / f"{slug}_benchq_{cfg['emb_key']}.npy"]
    if not all(p.exists() for p in paths):
        subprocess.run([sys.executable, str(ROOT / "scripts/embed_bench.py"), cfg["model"], cfg["emb_key"]], check=True)
    c_emb, q_emb = (np.load(p) for p in paths)
    assert len(c_emb) == len(corpus) and len(q_emb) == len(bq)
    return c_emb, q_emb


def main(cfg):
    t0 = time.time()
    train, bq, corpus = prepare(load("train")), prepare(load("benchmark_queries")), load("benchmark_items")
    cidx = {i: n for n, i in enumerate(corpus.item_id)}
    c_emb, q_emb = embeddings(cfg, corpus, bq)
    mem = ClickMemory(train, cidx)
    bm = BM25Source(bm25_item_text(corpus).tolist())
    boost = Booster(corpus, mem.item_pop, location_centroids(corpus, train), **cfg["boost"])
    boost.prepare(bq)
    fs = FullScorer(bm, c_emb, boost)
    top, feats = fs.run(dense_query_text(bq).tolist(), q_emb, K=cfg["pool"], w_bm25=cfg["w_bm25"], w_dense=cfg["w_dense"],
                        tau=cfg["tau"], boost_all=boost.all, details=True)
    if cfg["rerank"]:
        mc = MicrocatClassifier().fit(train)
        fb = FeatureBuilder(corpus, boost, mc.classes)
        X = fb.build(bq, top, feats, bq.q.isin(set(train.q)).values, mc.predict_proba(bq.q.tolist()), mem)
        top = pickle.load(open(ART / cfg["rerank"], "rb")).rerank(X, top, K)
    ans = build_answer(bq.query_id, [[corpus.item_id.iat[i] for i in f[:K]] for f in top])
    out = ROOT / f"answer_{cfg['name']}.csv"
    save(ans, out, bq.query_id, corpus.item_id)
    shutil.copy(out, ROOT / "answer.csv")
    with open(ART / "experiments.jsonl", "a") as f:
        f.write(json.dumps({"name": cfg["name"], "config": cfg, "time_s": round(time.time() - t0)}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(CONFIGS[sys.argv[1] if len(sys.argv) > 1 else "v4"])
