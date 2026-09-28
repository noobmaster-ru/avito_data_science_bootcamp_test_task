import numpy as np
import pandas as pd


def _xy(lat, lon) -> np.ndarray:
    """Координаты в км (плоская проекция для широт РФ)."""
    return np.column_stack([lat * 111.0, lon * 111.0 * np.cos(np.radians(55.0))])


class Booster:
    """Множители к скору кандидата: совпадение локации, расстояние до центроида локации запроса, популярность в логе, микрокатегория."""

    def __init__(self, corpus: pd.DataFrame, item_pop: dict, centroids: pd.DataFrame, mc_classes=None,
                 a: float = 0.0, b: float = 2.0, c: float = 0.5, d: float = 0.0, scale_km: float = 30.0):
        self.item_mc = corpus.item_microcat_id.fillna(-1).astype(int).values
        self.item_loc = corpus.item_location_id.fillna(-1).astype(int).values
        self.item_xy = _xy(pd.to_numeric(corpus.item_latitude, errors="coerce").values,
                           pd.to_numeric(corpus.item_longitude, errors="coerce").values)
        self.item_pop = np.zeros(len(corpus), np.float32)
        for i, n in item_pop.items():
            self.item_pop[i] = n
        self.log_pop = np.log1p(self.item_pop)
        self.centroids = centroids
        self.mc_col = {int(m): j for j, m in enumerate(mc_classes)} if mc_classes is not None else {}
        self.a, self.b, self.c, self.d, self.scale = a, b, c, d, scale_km

    def prepare(self, queries: pd.DataFrame, proba: np.ndarray | None = None):
        """Сохраняет для запросов локации, центроиды и (опционально) вероятности микрокатегорий."""
        self.proba = proba
        self.q_loc = queries.search_location_id.fillna(-1).astype(int).values
        c = self.centroids.reindex(self.q_loc)
        self.q_xy = _xy(c.lat.values, c.lon.values)

    def all(self, qi: int) -> np.ndarray:
        """Множители для всего корпуса (векторизовано)."""
        m = 1 + self.b * (self.item_loc == self.q_loc[qi])
        m = m * (1 + self.c * self.log_pop)
        if self.d:
            dist = np.linalg.norm(self.item_xy - self.q_xy[qi], axis=1)
            m = m * (1 + self.d * np.nan_to_num(np.exp(-dist / self.scale)))
        return m

    def __call__(self, qi: int, items: np.ndarray) -> np.ndarray:
        m = 1 + self.b * (self.item_loc[items] == self.q_loc[qi])
        m = m * (1 + self.c * np.log1p(self.item_pop[items]))
        if self.d:
            dist = np.linalg.norm(self.item_xy[items] - self.q_xy[qi], axis=1)
            m = m * (1 + self.d * np.nan_to_num(np.exp(-dist / self.scale)))
        if self.a and self.proba is not None:
            cols = np.array([self.mc_col.get(x, -1) for x in self.item_mc[items]])
            m = m * (1 + self.a * np.where(cols >= 0, self.proba[qi][np.maximum(cols, 0)], 0.0))
        return m


def _median_xy(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Медианные координаты объявлений, сгруппированные по колонке key."""
    df = df.assign(lat=pd.to_numeric(df.item_latitude, errors="coerce"), lon=pd.to_numeric(df.item_longitude, errors="coerce"))
    df = df.dropna(subset=["lat", "lon"])
    return df.groupby(df[key].fillna(-1).astype(int))[["lat", "lon"]].median()


def location_centroids(corpus: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    """Центроид локации запроса: медиана координат объявлений, кликнутых из этой локации поиска (log);
    для локаций, которых нет в логе, медиана координат объявлений с таким item_location_id (корпус + лог)."""
    by_search = _median_xy(log, "search_location_id")
    by_item = _median_xy(pd.concat([corpus, log[corpus.columns.intersection(log.columns)]], ignore_index=True), "item_location_id")
    return pd.concat([by_search, by_item[~by_item.index.isin(by_search.index)]])
