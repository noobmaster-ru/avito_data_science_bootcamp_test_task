import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from ..text import norm


class CharTfidfSource:
    """Символьный TF-IDF (char_wb 3–5) по коротким текстам объявлений; устойчив к опечаткам и морфологии."""

    def __init__(self, texts, ngram=(3, 5), min_df: int = 3, max_features: int = 400_000):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, min_df=min_df, max_features=max_features,
                                   sublinear_tf=True, dtype=np.float32)
        self.M = self.vec.fit_transform([norm(t) for t in texts]).T.tocsr()

    def retrieve(self, queries, k: int = 300, chunk: int = 256):
        """Топ-k по косинусу, чанками запросов; нулевые скоры → -1."""
        Q = self.vec.transform([norm(q) for q in queries])
        idx_all, sc_all = [], []
        for i in range(0, Q.shape[0], chunk):
            s = (Q[i:i + chunk] @ self.M).toarray()
            idx = np.argpartition(-s, k, axis=1)[:, :k]
            sc = np.take_along_axis(s, idx, 1)
            order = np.argsort(-sc, axis=1, kind="stable")
            idx, sc = np.take_along_axis(idx, order, 1), np.take_along_axis(sc, order, 1)
            idx_all.append(np.where(sc > 0, idx, -1))
            sc_all.append(sc)
        return np.vstack(idx_all), np.vstack(sc_all)
