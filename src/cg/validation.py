import numpy as np
import pandas as pd
from .config import SEED
from .text import norm, stem_text


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет нормализованный (q) и стеммированный (qs) текст запроса."""
    df = df.copy()
    df["q"] = df.search_query.map(norm)
    df["qs"] = df.q.map(stem_text)
    return df


def make_split(train: pd.DataFrame, n_val: int = 8000, seen_share: float = 0.375, seed: int = SEED):
    """V: по одной строке на уникальный текст запроса, доля «виденных» в A текстов как в бенчмарке; A: остальное."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"row": np.arange(len(train)), "q": train.q.values, "cnt": train.q.map(train.q.value_counts()).values})
    seen_pool = df[df.cnt >= 2].drop_duplicates("q").row.values
    unseen_pool = df[df.cnt == 1].row.values
    n_seen = int(n_val * seen_share)
    v = np.concatenate([rng.choice(seen_pool, n_seen, replace=False), rng.choice(unseen_pool, n_val - n_seen, replace=False)])
    mask = np.zeros(len(train), bool)
    mask[v] = True
    return train[~mask].reset_index(drop=True), train[mask].reset_index(drop=True)


def make_val_corpus(bench_items: pd.DataFrame, val: pd.DataFrame, train_cols: list[str]) -> pd.DataFrame:
    """Корпус валидации: корпус бенчмарка плюс релевантные объявления V (те же дистракторы, что в задаче)."""
    extra = val[train_cols].drop_duplicates("item_id")
    extra = extra[~extra.item_id.isin(set(bench_items.item_id))]
    return pd.concat([bench_items, extra[bench_items.columns]], ignore_index=True)
