import pandas as pd
from .config import DATA

ID_COLS = ["query_id", "item_id"]
INT_COLS = ["search_location_id", "search_category", "search_is_delivery_search", "item_category_id",
            "item_microcat_id", "item_location_id", "item_is_phone_hidden", "item_is_message_forbidden"]
TEXT_COLS = ["search_query", "search_infm_params_text", "item_title_raw", "item_description_raw", "item_infm_params_text"]


def load(name: str) -> pd.DataFrame:
    """Читает parquet: id строками, категории/локации/флаги как Int64, тексты без NaN."""
    df = pd.read_parquet(DATA / f"{name}.parquet")
    for c in ID_COLS:
        if c in df:
            df[c] = df[c].astype(str)
    for c in INT_COLS:
        if c in df and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].round().astype("Int64")
    for c in TEXT_COLS:
        if c in df:
            df[c] = df[c].fillna("").astype(str)
    return df


def item_text(df: pd.DataFrame, desc_len: int = 300) -> pd.Series:
    """Текст объявления для индексов: заголовок, параметры, начало описания."""
    return (df.item_title_raw + " | " + df.item_infm_params_text + " | "
            + df.item_description_raw.str.slice(0, desc_len)).str.strip(" |")


def query_text(df: pd.DataFrame) -> pd.Series:
    """Текст запроса для индексов: сам запрос плюс текстовые фильтры."""
    return (df.search_query + " " + df.search_infm_params_text).str.strip()
