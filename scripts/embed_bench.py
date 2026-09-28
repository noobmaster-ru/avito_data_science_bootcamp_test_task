"""Эмбеддинги корпуса и запросов бенчмарка заданной моделью (отдельный процесс: torch и LightGBM не уживаются в одном)."""
import sys
from cg.config import DENSE_MODEL
from cg.data import load
from cg.pipeline import dense_item_text, dense_query_text
from cg.sources.dense import DenseSource
model, key = (sys.argv + [DENSE_MODEL, "benchcorpus_v1"])[1:3]
ds = DenseSource(model)
ds.encode(dense_item_text(load("benchmark_items")).tolist(), "passage: ", key, batch=256)
ds.encode(dense_query_text(load("benchmark_queries")).tolist(), "query: ", "benchq_" + key, batch=256)
