import numpy as np
import pandas as pd
import lightgbm as lgb
from .text import stem_tokens

FEATS = ["rank", "bm25_norm", "dense_cos", "dense_exp", "boost", "score", "same_loc", "dist_km", "log_pop", "rating", "reviews",
         "log_price", "phone_hidden", "msg_forbidden", "cat_main", "title_len", "q_len", "q_ntok", "has_params", "seen",
         "mc_prob", "tok_overlap", "title_has_q", "mem_level", "mem_logn", "params_overlap"]


def title_stems(corpus: pd.DataFrame) -> list[set]:
    """Множества стемов заголовков объявлений."""
    return [set(stem_tokens(t)) for t in corpus.item_title_raw.values]


def params_stems(corpus: pd.DataFrame, max_chars: int = 1500) -> list[set]:
    """Множества стемов параметров объявлений (начало текста параметров)."""
    return [set(stem_tokens(t[:max_chars])) for t in corpus.item_infm_params_text.values]


class FeatureBuilder:
    """Строит таблицу признаков (запрос, кандидат) из пула полного скоринга и атрибутов объявлений."""

    def __init__(self, corpus: pd.DataFrame, booster, mc_classes=None):
        self.bo = booster
        self.stems = title_stems(corpus)
        self.pstems = params_stems(corpus)
        self.rating = pd.to_numeric(corpus.item_rating, errors="coerce").values
        self.reviews = pd.to_numeric(corpus.item_rating_reviews_count, errors="coerce").values
        self.log_price = np.log1p(pd.to_numeric(corpus.item_price, errors="coerce").clip(lower=0).values)
        self.phone = corpus.item_is_phone_hidden.fillna(0).astype(int).values
        self.msg = corpus.item_is_message_forbidden.fillna(0).astype(int).values
        self.cat_main = (corpus.item_category_id.fillna(-1).astype(int).values == 114).astype(int)
        self.title_len = corpus.item_title_raw.str.len().values
        self.titles_norm = corpus.item_title_raw.str.lower().values
        self.mc_col = {int(m): j for j, m in enumerate(mc_classes)} if mc_classes is not None else {}

    def build(self, queries: pd.DataFrame, top: list, feats: list, seen: np.ndarray, proba=None, mem=None) -> pd.DataFrame:
        """Таблица признаков; строки идут по запросам подряд, размер группы = len(top[qi])."""
        rows = []
        found = mem.lookup(queries) if mem is not None else [{} for _ in range(len(queries))]
        q_stems = [set(stem_tokens(q)) for q in queries.q.values]
        p_stems = [set(stem_tokens(p)) - {"вид", "тип", "услуг"} for p in queries.search_infm_params_text.values]
        q_text = queries.q.values
        has_params = (queries.search_infm_params_text != "").astype(int).values
        for qi, (items, f) in enumerate(zip(top, feats)):
            it = np.asarray(items)
            n = len(it)
            same = (self.bo.item_loc[it] == self.bo.q_loc[qi]).astype(np.float32)
            dist = np.nan_to_num(np.linalg.norm(self.bo.item_xy[it] - self.bo.q_xy[qi], axis=1), nan=5000.0)
            qs = q_stems[qi]
            overlap = np.array([len(qs & self.stems[i]) / max(len(qs), 1) for i in it], np.float32)
            has_q = np.array([q_text[qi] in self.titles_norm[i] for i in it], np.float32)
            if proba is not None and self.mc_col:
                cols = np.array([self.mc_col.get(m, -1) for m in self.bo.item_mc[it]])
                mcp = np.where(cols >= 0, proba[qi][np.maximum(cols, 0)], 0.0)
            else:
                mcp = np.zeros(n, np.float32)
            ps = p_stems[qi]
            povl = np.array([len(ps & self.pstems[i]) / len(ps) for i in it], np.float32) if ps else np.zeros(n, np.float32)
            ml = np.array([found[qi].get(i, (4, 0))[0] for i in it], np.float32)
            mn = np.log1p(np.array([found[qi].get(i, (4, 0))[1] for i in it], np.float32))
            rows.append(np.column_stack([
                np.arange(n), f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4], same, dist, self.bo.log_pop[it],
                self.rating[it], self.reviews[it], self.log_price[it], self.phone[it], self.msg[it], self.cat_main[it],
                self.title_len[it], np.full(n, len(q_text[qi])), np.full(n, len(qs)), np.full(n, has_params[qi]),
                np.full(n, int(seen[qi])), mcp, overlap, has_q, ml, mn, povl]))
        return pd.DataFrame(np.vstack(rows), columns=FEATS)


class Reranker:
    """LightGBM-классификатор «кандидат релевантен»; rerank переупорядочивает пул и берёт топ-K."""

    def __init__(self, **params):
        self.params = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=8) | params
        self.model = lgb.LGBMClassifier(**self.params)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.model.fit(X[FEATS], y)
        return self

    def rerank(self, X: pd.DataFrame, top: list, K: int = 50) -> list:
        """Пересортировка кандидатов каждого запроса по вероятности релевантности."""
        p = self.model.predict_proba(X[FEATS])[:, 1]
        out, pos = [], 0
        for items in top:
            n = len(items)
            order = np.lexsort((np.arange(n), -p[pos:pos + n]))
            out.append([items[i] for i in order[:K]])
            pos += n
        return out
