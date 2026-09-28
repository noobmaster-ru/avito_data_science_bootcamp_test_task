"""Генерирует notebooks/02_experiments.ipynb: путь от бейзлайна к финальному пайплайну на локальной валидации."""
import nbformat as nbf
nb = nbf.v4.new_notebook(); c = nb.cells
md = lambda s: c.append(nbf.v4.new_markdown_cell(s)); code = lambda s: c.append(nbf.v4.new_code_cell(s))
md("""# 02. Эксперименты на локальной валидации

Все числа ниже считаются на одном и том же сплите (`artifacts/val_split.pkl`, строится `cg.validation.make_split`):
8 000 запросов с уникальными текстами, 37,5 % из них встречаются в обучающей части A, корпус = корпус бенчмарка + релевантные
объявления валидации (196 518 объявлений). Метрика та же, что в задаче: Recall@50, усреднённый по запросам.
Кандидаты источников закэшированы скриптами из `scripts/`, здесь они только объединяются и оцениваются.""")
code("""import sys; sys.path.insert(0, "../src")
import pickle, numpy as np, pandas as pd
from cg.config import ART
from cg.metrics import report
from cg.fusion import fuse
from cg.sources.memory import ClickMemory
from cg.boost import Booster, location_centroids
d = pickle.load(open(ART / "val_split.pkl", "rb"))
A, V, corpus, rel, seen_q = d["A"], d["V"], d["corpus"], d["rel"], d["seen_q"]
c1 = pickle.load(open(ART / "val_cands_v1.pkl", "rb"))        # память, добор по микрокатегориям, BM25 top-300
bm5k = pickle.load(open(ART / "val_cands_bm5000.pkl", "rb"))  # BM25 top-5000
dense = pickle.load(open(ART / "val_cands_dense.pkl", "rb"))  # e5-small top-1000
cidx = {i: n for n, i in enumerate(corpus.item_id)}
mem = ClickMemory(A, cidx); cent = location_centroids(corpus, A)
print(len(A), len(V), len(corpus))""")
md("## 1. Отдельные источники\n\nПамять кликов почти бесполезна (мало пересечений с корпусом), BM25 и dense по отдельности дают около 0,4.")
code("""print("память кликов:      ", report(c1["mem"], rel, seen_q))
print("добор по микрокат.: ", report(c1["prior"], rel, seen_q))
print("BM25 top-300:       ", report(c1["bm25"].tolist(), rel, seen_q))
print("e5-small top-1000:  ", report(dense["dense_1000"].tolist(), rel, seen_q, ks=(50, 300, 1000)))""")
md("""## 2. Бусты по локации, расстоянию и популярности

Скор кандидата умножается на `(1 + b*[та же локация]) * (1 + d*exp(-dist/scale)) * (1 + c*log(1 + клики в train))`.
Локация даёт главный прирост: с 0,40 до 0,60 при том же пуле из 300 кандидатов.""")
code("""def run(pool_k, **bk):
    bo = Booster(corpus, mem.item_pop, cent, **bk); bo.prepare(V)
    return report(fuse({"bm25": bm5k["bm25_5000"][:, :pool_k].tolist()}, {"bm25": 1.0}, forced=c1["mem"], quota=5, filler=c1["prior"], boost=bo), rel, seen_q)
print("пул 300, без бустов:          ", run(300, b=0, c=0, d=0))
print("пул 300, локация:             ", run(300, b=5, c=0, d=0))
print("пул 300, локация+расст.+попул:", run(300, b=5, c=0.5, d=5, scale_km=100))""")
md("## 3. Размер пула кандидатов\n\nС бустами качество упирается в полноту пула BM25, поэтому пул расширяется, а в финале скорится весь корпус.")
code("""for k in (300, 1000, 2000, 5000):
    print(f"пул {k}: ", run(k, b=5, c=0.5, d=5, scale_km=100)["r@50"])""")
md("""## 4. Полный скоринг корпуса, BM25 + dense, реранкер

Дальше кандидаты не ограничиваются пулом: для каждого запроса считаются BM25 и косинус e5 по всем объявлениям
(`cg.scoring.FullScorer`), скоры нормируются в (0, 1] и складываются, затем применяются бусты.
Результаты на случайной подвыборке из 2 500 запросов валидации (`scripts/val_tune.py`, `scripts/val_full.py`):

| Вариант | Recall@50 |
|---|---|
| BM25 по всему корпусу + бусты | 0,79 |
| e5-small по всему корпусу + бусты | 0,78 |
| BM25 + e5, tau = 10 / 20 / 50 / 100 / 200 | 0,79 / 0,82 / **0,82** / 0,81 / 0,79 |
| RRF по рангам вместо суммы скоров | 0,82 |

Топ-300 полного скоринга (Recall@300 = 0,92) переупорядочивается LightGBM-классификатором на 23 признаках
(`cg.rerank.FEATS`): 2 фолда по запросам, `scripts/rerank_train.py`.""")
code("""pool = pickle.load(open(ART / "val_rerank_pool.pkl", "rb"))
print("пул полного скоринга:", report(pool["top"], rel, seen_q, ks=(50, 100, 300)))
print(open(ART / "rerank_train.log").read().split("pool r@50/300")[1][:1200])""")
nbf.write(nb, "notebooks/02_experiments.ipynb")
