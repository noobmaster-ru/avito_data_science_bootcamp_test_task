import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from ..text import norm


class MicrocatClassifier:
    """Текст запроса → распределение по микрокатегориям: TF-IDF (слова + символы) и логистическая регрессия SGD."""

    def __init__(self, alpha: float = 2e-6, epochs: int = 15):
        self.vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, dtype=np.float32)
        self.vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True, dtype=np.float32)
        self.clf = SGDClassifier(loss="log_loss", alpha=alpha, max_iter=epochs, tol=None, n_jobs=8, random_state=0)

    def _x(self, texts, fit: bool = False):
        t = [norm(s) for s in texts]
        return hstack([self.vw.fit_transform(t) if fit else self.vw.transform(t),
                       self.vc.fit_transform(t) if fit else self.vc.transform(t)]).tocsr()

    def fit(self, log: pd.DataFrame):
        """Обучение на уникальных парах (текст запроса, микрокатегория) с весом = число кликов."""
        g = log.groupby(["q", "item_microcat_id"], observed=True).size().reset_index(name="n")
        self.clf.fit(self._x(g.q.tolist(), fit=True), g.item_microcat_id.astype(int).values, sample_weight=g.n.values)
        self.classes = self.clf.classes_
        return self

    def predict_proba(self, texts) -> np.ndarray:
        """Матрица вероятностей (n_queries × n_classes)."""
        return self.clf.predict_proba(self._x(texts))
