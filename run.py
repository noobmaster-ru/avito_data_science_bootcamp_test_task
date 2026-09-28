"""Сборка ответа для бенчмарка: полный скоринг корпуса (BM25 + bi-encoder'ы + бусты) → реранкер → submissions/answer.csv.
Запуск: python run.py [final|no_finetune|no_rerank]. Эмбеддинги кэшируются в artifacts/emb, при их отсутствии считаются;
веса реранкера берутся из release/ (в git), а при их отсутствии из artifacts/ (результат scripts/07_rerank_train.py)."""
import hashlib, json, pickle, subprocess, sys, time
import numpy as np
from cg.config import ART, RELEASE, ROOT, K, DENSE_MODEL, FT_MODEL
from cg.data import load
from cg.validation import prepare
from cg.pipeline import bm25_item_text, dense_query_text
from cg.sources.memory import ClickMemory
from cg.sources.bm25 import BM25Source
from cg.sources.microcat import MicrocatClassifier
from cg.boost import Booster, location_centroids
from cg.scoring import MultiScorer
from cg.rerank import FeatureBuilder
from cg.submit import build_answer, save

BASE = (DENSE_MODEL, "benchcorpus_v1", 50.0, 1.0)   # (модель, ключ кэша эмбеддингов, tau, вес)
FT = (FT_MODEL, "benchcorpus_ft1", 10.0, 2.0)
BOOST = dict(b=5, c=0.5, d=5, scale_km=100)
CONFIGS = {
    "final": dict(dense=[BASE, FT], pool=500, rerank="reranker_final.pkl", out="answer.csv"),
    "no_finetune": dict(dense=[BASE], pool=300, rerank="reranker_single_encoder.pkl", out="answer_no_finetune.csv"),
    "no_rerank": dict(dense=[BASE, FT], pool=K, rerank=None, out="answer_no_rerank.csv"),
}


def embeddings(model: str, key: str, corpus, bq):
    """Эмбеддинги корпуса и запросов бенчмарка из кэша; если кэша нет, считает их в подпроцессе (torch отдельно от LightGBM)."""
    slug = model.replace("/", "_")
    paths = [ART / "emb" / f"{slug}_{key}.npy", ART / "emb" / f"{slug}_benchq_{key}.npy"]
    if not all(p.exists() for p in paths):
        subprocess.run([sys.executable, str(ROOT / "scripts" / "embed_bench.py"), model, key], check=True)
    c_emb, q_emb = (np.load(p) for p in paths)
    assert len(c_emb) == len(corpus) and len(q_emb) == len(bq)
    return c_emb, q_emb


def main(name: str):
    cfg, t0 = CONFIGS[name], time.time()
    ART.mkdir(exist_ok=True)
    train, bq, corpus = prepare(load("train")), prepare(load("benchmark_queries")), load("benchmark_items")
    cidx = {i: n for n, i in enumerate(corpus.item_id)}
    embs = [embeddings(model, key, corpus, bq) for model, key, tau, w in cfg["dense"]]
    mem = ClickMemory(train, cidx)
    bm = BM25Source(bm25_item_text(corpus).tolist())
    boost = Booster(corpus, mem.item_pop, location_centroids(corpus, train), **BOOST)
    boost.prepare(bq)
    scorer = MultiScorer(bm, [(c, tau, w) for (c, q), (_, _, tau, w) in zip(embs, cfg["dense"])], boost)
    top, feats = scorer.run(dense_query_text(bq).tolist(), [q for c, q in embs], K=cfg["pool"], boost_all=boost.all, details=True)
    if cfg["rerank"]:
        mc = MicrocatClassifier().fit(train)
        X = FeatureBuilder(corpus, boost, mc.classes).build(bq, top, feats, bq.q.isin(set(train.q)).values, mc.predict_proba(bq.q.tolist()), mem)
        path = RELEASE / cfg["rerank"] if (RELEASE / cfg["rerank"]).exists() else ART / cfg["rerank"]
        top = pickle.load(open(path, "rb")).rerank(X, top, K)
    ans = build_answer(bq.query_id, [[corpus.item_id.iat[i] for i in f[:K]] for f in top])
    out = ROOT / "submissions" / cfg["out"]
    out.parent.mkdir(exist_ok=True)
    save(ans, out, bq.query_id, corpus.item_id)
    print(f"md5 {hashlib.md5(out.read_bytes()).hexdigest()}  {out.relative_to(ROOT)}  ({round(time.time() - t0)} s)")
    with open(ART / "experiments.jsonl", "a") as f:
        f.write(json.dumps({"config": name, "out": str(out), "time_s": round(time.time() - t0)}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "final")
