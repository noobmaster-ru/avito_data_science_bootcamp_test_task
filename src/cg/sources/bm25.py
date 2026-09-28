import bm25s
import numpy as np
import Stemmer
from ..text import norm


class BM25Source:
    """BM25 (bm25s + Snowball) по тексту объявлений; retrieve отдаёт индексы и скоры (n_q × k)."""

    def __init__(self, texts, k1: float = 1.2, b: float = 0.75):
        self.stemmer = Stemmer.Stemmer("russian")
        toks = bm25s.tokenize([norm(t) for t in texts], stopwords=None, stemmer=self.stemmer, show_progress=False)
        self.index = bm25s.BM25(k1=k1, b=b)
        self.index.index(toks, show_progress=False)

    @staticmethod
    def _norm(q: str) -> str:
        """Нормализация запроса, пустой заменяется заглушкой."""
        return norm(q) or "пусто"

    def retrieve(self, queries, k: int = 300, n_threads: int = 8):
        """Топ-k по BM25; кандидаты с нулевым скором заменяются на -1."""
        qs = [norm(q) or "пусто" for q in queries]
        toks = bm25s.tokenize(qs, stopwords=None, stemmer=self.stemmer, show_progress=False, return_ids=False)
        toks = [t or ["пусто"] for t in toks]
        docs, scores = self.index.retrieve(toks, k=k, show_progress=False, n_threads=n_threads)
        docs = np.where(scores > 0, docs, -1)
        return docs, scores
