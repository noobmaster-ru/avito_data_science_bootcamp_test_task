import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from ..config import ART, DENSE_MODEL


def device() -> str:
    """MPS на Apple Silicon, иначе CPU."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


class DenseSource:
    """Bi-encoder (e5): эмбеддинги с кэшем на диске, поиск по косинусу через numpy."""

    def __init__(self, model_name: str = DENSE_MODEL, max_len: int = 128):
        self.model = SentenceTransformer(model_name, device=device())
        self.model.max_seq_length = max_len
        self.slug = model_name.replace("/", "_")

    def encode(self, texts, prefix: str, cache_key: str | None = None, batch: int = 128) -> np.ndarray:
        """Нормированные эмбеддинги fp16; при cache_key читает/пишет artifacts/emb/<slug>_<key>.npy."""
        path = ART / "emb" / f"{self.slug}_{cache_key}.npy" if cache_key else None
        if path is not None and path.exists():
            return np.load(path)
        emb = self.model.encode([prefix + t for t in texts], batch_size=batch, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=True).astype(np.float16)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, emb)
        return emb


def topk(q_emb: np.ndarray, c_emb: np.ndarray, k: int = 300, chunk: int = 1024):
    """Топ-k по скалярному произведению нормированных векторов, чанками запросов."""
    c = np.ascontiguousarray(c_emb.T.astype(np.float32))
    idx_all, sc_all = [], []
    for i in range(0, len(q_emb), chunk):
        s = q_emb[i:i + chunk].astype(np.float32) @ c
        idx = np.argpartition(-s, k, axis=1)[:, :k]
        sc = np.take_along_axis(s, idx, 1)
        order = np.argsort(-sc, axis=1, kind="stable")
        idx_all.append(np.take_along_axis(idx, order, 1))
        sc_all.append(np.take_along_axis(sc, order, 1))
    return np.vstack(idx_all), np.vstack(sc_all)
