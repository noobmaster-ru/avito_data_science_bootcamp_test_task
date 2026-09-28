"""Проверка формата файла ответа: колонки, все query_id бенчмарка, до 50 уникальных item_id из корпуса, строковые id."""
import sys, pandas as pd
from cg.config import ROOT
from cg.data import load
from cg.submit import validate
path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "submissions" / "answer.csv"
ans = pd.read_csv(path, dtype=str, keep_default_na=False)
validate(ans, load("benchmark_queries").query_id, load("benchmark_items").item_id)
print(f"OK: {path} — {len(ans)} строк, в среднем {ans.answer.str.split().str.len().mean():.1f} item_id на запрос")
