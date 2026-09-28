import time, pickle, numpy as np, pandas as pd
from cg.data import load, item_text, query_text
from cg.validation import prepare, make_split, make_val_corpus
from cg.sources.memory import ClickMemory, MicrocatPrior
from cg.sources.bm25 import BM25Source
from cg.metrics import report
from cg.fusion import fuse
from cg.config import ART
t0 = time.time()
tr = prepare(load("train")); bi = load("benchmark_items")
A, V = make_split(tr)
corpus = make_val_corpus(bi, V, [c for c in tr.columns if c.startswith("item_")])
cidx = {i: n for n, i in enumerate(corpus.item_id)}
rel = [[cidx[i]] for i in V.item_id]
seen_q = V.q.isin(set(A.q)).values
seen_item = V.item_id.isin(set(A.item_id)).values
print(f"A {len(A)} V {len(V)} corpus {len(corpus)} seen_q {seen_q.mean():.3f} seen_item {seen_item.mean():.3f} t={time.time()-t0:.0f}s")
pickle.dump(dict(A=A, V=V, corpus=corpus, rel=rel, seen_q=seen_q, seen_item=seen_item), open(ART/"val_split.pkl", "wb"))
mem = ClickMemory(A, cidx)
mem_lists, lvl = mem.retrieve(V, 50)
print("memory found level dist", pd.Series(lvl).value_counts().to_dict())
print("memory:", report(mem_lists, rel, seen_q), f"t={time.time()-t0:.0f}s")
prior = MicrocatPrior(A, corpus, mem.item_pop)
prior_lists = prior.retrieve(V, 300)
print("microcat prior:", report(prior_lists, rel, seen_q), f"t={time.time()-t0:.0f}s")
texts = (corpus.item_title_raw + " . ") * 3 + corpus.item_infm_params_text.str.slice(0, 600) + " . " + corpus.item_description_raw.str.slice(0, 300)
bm = BM25Source(texts.tolist())
print(f"bm25 indexed t={time.time()-t0:.0f}s")
bm_idx, bm_sc = bm.retrieve(query_text(V).tolist(), 300)
print("bm25 q+params:", report(bm_idx.tolist(), rel, seen_q), f"t={time.time()-t0:.0f}s")
bm_idx2, _ = bm.retrieve(V.search_query.tolist(), 300)
print("bm25 q only:", report(bm_idx2.tolist(), rel, seen_q))
fused = fuse({"bm25": bm_idx.tolist()}, {"bm25": 1.0}, forced=mem_lists, quota=20, filler=prior_lists)
print("fused mem+bm25+prior:", report(fused, rel, seen_q))
pickle.dump(dict(mem=mem_lists, prior=prior_lists, bm25=bm_idx, bm25_sc=bm_sc), open(ART/"val_cands_v1.pkl", "wb"))
print("done", f"t={time.time()-t0:.0f}s")
