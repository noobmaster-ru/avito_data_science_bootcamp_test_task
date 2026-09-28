import re
import pandas as pd

_ITEM_RE = re.compile(r"^[0-9a-f]{16}$")


def build_answer(query_ids, preds) -> pd.DataFrame:
    """Собирает answer.csv: query_id и до 50 item_id через пробел."""
    return pd.DataFrame({"query_id": list(query_ids), "answer": [" ".join(p[:50]) for p in preds]})


def validate(ans: pd.DataFrame, bench_query_ids, corpus_item_ids, k: int = 50) -> None:
    """Строгая проверка формата ответа; бросает AssertionError с описанием проблемы."""
    assert list(ans.columns) == ["query_id", "answer"], f"колонки: {list(ans.columns)}"
    assert ans.query_id.is_unique, "дубли query_id"
    assert set(ans.query_id) == set(bench_query_ids), "набор query_id не совпадает с бенчмарком"
    corpus = set(corpus_item_ids)
    lens = []
    for q, a in zip(ans.query_id, ans.answer):
        items = a.split(" ")
        assert 1 <= len(items) <= k, f"{q}: {len(items)} item_id"
        assert len(set(items)) == len(items), f"{q}: дубли item_id"
        assert all(_ITEM_RE.match(i) for i in items), f"{q}: плохой формат item_id"
        assert all(i in corpus for i in items), f"{q}: item_id вне корпуса"
        lens.append(len(items))
    mean_len = sum(lens) / len(lens)
    if mean_len < k:
        print(f"warning: средняя длина ответа {mean_len:.1f} < {k}")


def save(ans: pd.DataFrame, path, bench_query_ids, corpus_item_ids) -> None:
    """Пишет csv без индекса, перечитывает строками и валидирует."""
    ans.to_csv(path, index=False, encoding="utf-8")
    back = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert back.equals(ans.astype(str)), "файл после перечитывания отличается"
    validate(back, bench_query_ids, corpus_item_ids)
    print(f"saved {path}: {len(back)} rows")
