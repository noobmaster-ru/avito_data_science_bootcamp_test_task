import numpy as np
import bm25s


class FullScorer:
    """Скоринг всего корпуса на запрос: BM25 и dense по всем документам, комбинация, множители-бусты, топ-K."""

    def __init__(self, bm25=None, c_emb: np.ndarray | None = None, boost=None):
        self.bm25 = bm25
        self.c = np.ascontiguousarray(c_emb.T.astype(np.float32)) if c_emb is not None else None
        self.boost = boost

    def _bm25_scores(self, tokens: list[str]) -> np.ndarray:
        vocab = self.bm25.index.vocab_dict
        toks = [t for t in tokens if t in vocab]
        return self.bm25.index.get_scores(toks) if toks else np.zeros(self.bm25.index.scores["num_docs"], np.float32)

    def run(self, queries_text, q_emb=None, K: int = 50, w_bm25: float = 1.0, w_dense: float = 1.0, tau: float = 20.0,
            mode: str = "score", boost_all=None, details: bool = False) -> list:
        """Для каждого запроса топ-K индексов корпуса; mode=score: bm25/max + exp(tau*(cos-max)); mode=rrf: по рангам.
        details=True: дополнительно возвращает по запросу матрицу признаков (bm25_norm, dense_cos, dense_exp, boost, score)."""
        toks = None
        if self.bm25 is not None and w_bm25:
            toks = bm25s.tokenize([self.bm25._norm(q) for q in queries_text], stopwords=None, stemmer=self.bm25.stemmer,
                                  show_progress=False, return_ids=False)
        out, feats = [], []
        for qi in range(len(queries_text)):
            s, b, cos, dsc = None, None, None, None
            if toks is not None:
                b = self._bm25_scores(toks[qi]).astype(np.float32)
                if mode == "rrf":
                    b = 1.0 / (60 + np.argsort(np.argsort(-b, kind="stable")))
                else:
                    b = b / max(b.max(), 1e-6)
                s = w_bm25 * b
            if self.c is not None and w_dense:
                cos = q_emb[qi].astype(np.float32) @ self.c
                if mode == "rrf":
                    dsc = 1.0 / (60 + np.argsort(np.argsort(-cos, kind="stable")))
                else:
                    dsc = np.exp(tau * (cos - cos.max()))
                s = w_dense * dsc if s is None else s + w_dense * dsc
            m = boost_all(qi) if boost_all is not None else None
            if m is not None:
                s = s * m
            idx = np.argpartition(-s, K)[:K]
            idx = idx[np.lexsort((idx, -s[idx]))]
            out.append(idx.tolist())
            if details:
                cols = [b[idx] if b is not None else np.zeros(K), cos[idx] if cos is not None else np.zeros(K),
                        dsc[idx] if dsc is not None else np.zeros(K), m[idx] if m is not None else np.ones(K), s[idx]]
                feats.append(np.column_stack(cols).astype(np.float32))
        return (out, feats) if details else out
