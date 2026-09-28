import numpy as np
import pandas as pd

INT_KEY_COLS = ["search_location_id", "search_category", "search_is_delivery_search"]
LEVELS = [
    ["q", "search_location_id", "search_category", "search_infm_params_text", "search_is_delivery_search"],
    ["q", "search_location_id"],
    ["q"],
    ["qs"],
]


def _keys(df: pd.DataFrame, cols: list[str]) -> list:
    """Хэшируемые ключи по колонкам (Int64 NA → -1)."""
    vals = [df[c].fillna(-1).astype(int).values if c in INT_KEY_COLS else df[c].values for c in cols]
    return list(zip(*vals)) if len(cols) > 1 else list(vals[0])


class ClickMemory:
    """Память кликов: ключ запроса → объявления корпуса по убыванию частоты, backoff от точного ключа к стеммированному тексту."""

    def __init__(self, log: pd.DataFrame, corpus_index: dict, levels=LEVELS):
        log = log[log.item_id.isin(corpus_index)].copy()
        log["iidx"] = log.item_id.map(corpus_index).astype(int)
        self.levels, self.tables, self.counts = levels, [], []
        for cols in levels:
            g = log.groupby(cols + ["iidx"], observed=True, dropna=False).size().reset_index(name="n")
            g = g.sort_values(["n", "iidx"], ascending=[False, True], kind="stable")
            keys, iidx, n = _keys(g, cols), g.iidx.values, g.n.values
            table, counts = {}, {}
            for pos, key in enumerate(keys):
                table.setdefault(key, []).append(iidx[pos])
                counts.setdefault(key, []).append(n[pos])
            self.tables.append(table)
            self.counts.append(counts)
        self.item_pop = log.iidx.value_counts().to_dict()

    def retrieve(self, queries: pd.DataFrame, k: int = 50) -> tuple[list, list]:
        """Для каждого запроса список объявлений (до k) и уровень backoff, на котором нашлись первые."""
        level_keys = [_keys(queries, cols) for cols in self.levels]
        out, found_level = [], []
        for qi in range(len(queries)):
            res, seen, lvl = [], set(), -1
            for li, table in enumerate(self.tables):
                for it in table.get(level_keys[li][qi], ()):
                    if it not in seen:
                        seen.add(it)
                        res.append(it)
                        lvl = li if lvl < 0 else lvl
                if len(res) >= k:
                    break
            out.append(res[:k])
            found_level.append(lvl)
        return out, found_level


    def lookup(self, queries: pd.DataFrame) -> list[dict]:
        """Для каждого запроса словарь объявление → (уровень backoff, число кликов) по первому уровню, где оно нашлось."""
        level_keys = [_keys(queries, cols) for cols in self.levels]
        out = []
        for qi in range(len(queries)):
            found = {}
            for li, (table, counts) in enumerate(zip(self.tables, self.counts)):
                key = level_keys[li][qi]
                for it, n in zip(table.get(key, ()), counts.get(key, ())):
                    found.setdefault(it, (li, n))
            out.append(found)
        return out


class MicrocatPrior:
    """Добор: текст запроса → микрокатегории по доле кликов → популярные объявления корпуса в них."""

    def __init__(self, log: pd.DataFrame, corpus: pd.DataFrame, item_pop: dict, key: str = "qs"):
        g = log.groupby([key, "item_microcat_id"], observed=True, dropna=False).size().reset_index(name="n")
        g = g.sort_values("n", ascending=False, kind="stable")
        self.key, self.q2mc = key, {}
        for q, mc in zip(g[key].values, g.item_microcat_id.fillna(-1).astype(int).values):
            self.q2mc.setdefault(q, []).append(mc)
        c = corpus.assign(iidx=np.arange(len(corpus)), pop=np.arange(len(corpus)))
        c["pop"] = c.iidx.map(item_pop).fillna(0)
        c = c.sort_values(["pop", "item_rating_reviews_count"], ascending=False, kind="stable")
        self.mc2items = {mc: grp.iidx.values for mc, grp in c.groupby(c.item_microcat_id.fillna(-1).astype(int))}

    def retrieve(self, queries: pd.DataFrame, k: int = 50) -> list:
        """Список популярных объявлений из микрокатегорий запроса (до k)."""
        out = []
        for q in queries[self.key].values:
            res = []
            for mc in self.q2mc.get(q, ()):
                res.extend(self.mc2items.get(mc, ())[: k - len(res)])
                if len(res) >= k:
                    break
            out.append(res[:k])
        return out
