"""Эмбеддинги исходного multilingual-e5-small для запросов V, запросов бенчмарка и корпуса валидации (кэш в artifacts/emb)."""
import pickle, time, numpy as np
from cg.config import ART
from cg.data import load
from cg.pipeline import dense_item_text, dense_query_text
from cg.sources.dense import DenseSource
t0 = time.time()
d = pickle.load(open(ART/"val_split.pkl", "rb"))
corpus, V = d["corpus"], d["V"]
bq = load("benchmark_queries")
ds = DenseSource()
print("model loaded", time.time()-t0, flush=True)
ds.encode(dense_query_text(V).tolist(), "query: ", "valq_v1", batch=256)
ds.encode(dense_query_text(bq).tolist(), "query: ", "benchq_v1", batch=256)
print("queries done", time.time()-t0, flush=True)
ds.encode(dense_item_text(corpus).tolist(), "passage: ", "valcorpus_v1", batch=256)
print("corpus done", time.time()-t0, flush=True)
