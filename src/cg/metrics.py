import numpy as np


def recall_at_k(preds, relevant, k: int = 50) -> float:
    """Средняя по запросам доля релевантных объявлений, попавших в топ-k."""
    return float(np.mean([len(set(p[:k]) & set(r)) / len(r) for p, r in zip(preds, relevant)]))


def report(preds, relevant, seen=None, ks=(50, 100, 300)) -> dict:
    """Recall@k общий и в разрезе seen/unseen (текст запроса встречался в логе или нет)."""
    out = {f"r@{k}": round(recall_at_k(preds, relevant, k), 4) for k in ks}
    if seen is not None:
        seen = np.asarray(seen, bool)
        for name, m in (("seen", seen), ("unseen", ~seen)):
            if m.any():
                out[f"r@50_{name}"] = round(recall_at_k([p for p, f in zip(preds, m) if f],
                                                        [r for r, f in zip(relevant, m) if f], 50), 4)
        out["seen_share"] = round(float(seen.mean()), 3)
    return out
