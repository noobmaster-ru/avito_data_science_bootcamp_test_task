import numpy as np
import pandas as pd
from .config import SEED


def split_train(train: pd.DataFrame, n_val: int = 8000, corpus_ids=None, seed: int = SEED):
    """Случайные n_val строк → V (валидация, релевантный = объявление строки), остальное → A."""
    rng = np.random.default_rng(seed)
    cand = np.arange(len(train)) if corpus_ids is None else np.flatnonzero(train.item_id.isin(corpus_ids).values)
    v = rng.choice(cand, size=min(n_val, len(cand)), replace=False)
    mask = np.zeros(len(train), bool)
    mask[v] = True
    return train[~mask].reset_index(drop=True), train[mask].reset_index(drop=True)
