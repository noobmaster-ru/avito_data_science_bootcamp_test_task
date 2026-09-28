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
from cg.scoring import MultiScorer
from cg.rerank import FeatureBuilder
from cg.submit import build_answer, save

FT_MODEL = str(ART / "models/e5s_ft_v1")
CONFIGS = {
    "v7": dict(name="v7_rerank_ft_w2_pool500", dense=[(DENSE_MODEL, "benchcorpus_v1", 50.0, 1.0), (FT_MODEL, "benchcorpus_ft1", 10.0, 2.0)],
               boost=dict(b=5, c=0.5, d=5, scale_km=100), rerank="reranker_v7.pkl", pool=500),
    "v6": dict(name="v6_rerank_two_encoders", dense=[(DENSE_MODEL, "benchcorpus_v1", 50.0, 1.0), (FT_MODEL, "benchcorpus_ft1", 10.0, 1.0)],
               boost=dict(b=5, c=0.5, d=5, scale_km=100), rerank="reranker_v6.pkl", pool=300),
    "v3": dict(name="v3_full_bm25_dense_tau50", model=DENSE_MODEL, emb_key="benchcorpus_v1", w_bm25=1.0, w_dense=1.0, tau=50.0,
               boost=dict(b=5, c=0.5, d=5, scale_km=100), rerank=None, pool=50),
    "v4": dict(name="v4c_rerank_desc6000", model=DENSE_MODEL, emb_key="benchcorpus_v1", w_bm25=1.0, w_dense=1.0, tau=50.0,
               boost=dict(b=5, c=0.5, d=5, scale_km=100), rerank="reranker_v5.pkl", pool=300),
    "v5": dict(name="v5_rerank_ft", model=str(ART / "models/e5s_ft_v1"), emb_key="benchcorpus_ft1", w_bm25=1.0, w_dense=1.0, tau=50.0,
               boost=dict(b=5, c=0.5, d=5, scale_km=100), rerank="reranker_ft1.pkl", pool=300),
}


def embeddings(model, key, corpus, bq):
    """Эмбеддинги корпуса и запросов бенчмарка из кэша artifacts/emb; если кэша нет, считает их в подпроцессе."""
    slug = model.replace("/", "_")
    paths = [ART / "emb" / f"{slug}_{key}.npy", ART / "emb" / f"{slug}_benchq_{key}.npy"]
    if not all(p.exists() for p in paths):
        subprocess.run([sys.executable, str(ROOT / "scripts/embed_bench.py"), model, key], check=True)
    c_emb, q_emb = (np.load(p) for p in paths)
    assert len(c_emb) == len(corpus) and len(q_emb) == len(bq)
    return c_emb, q_emb


def main(cfg):
    t0 = time.time()
    train, bq, corpus = prepare(load("train")), prepare(load("benchmark_queries")), load("benchmark_items")
    cidx = {i: n for n, i in enumerate(corpus.item_id)}
    dense = cfg.get("dense") or [(cfg["model"], cfg["emb_key"], cfg["tau"], 1.0)]
    embs = [embeddings(model, key, corpus, bq) for model, key, tau, w in dense]
    mem = ClickMemory(train, cidx)
    bm = BM25Source(bm25_item_text(corpus).tolist())
    boost = Booster(corpus, mem.item_pop, location_centroids(corpus, train), **cfg["boost"])
    boost.prepare(bq)
    fs = MultiScorer(bm, [(c, tau, w) for (c, q), (_, _, tau, w) in zip(embs, dense)], boost)
    top, feats = fs.run(dense_query_text(bq).tolist(), [q for c, q in embs], K=cfg["pool"], boost_all=boost.all, details=True)
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
    main(CONFIGS[sys.argv[1] if len(sys.argv) > 1 else "v7"])
