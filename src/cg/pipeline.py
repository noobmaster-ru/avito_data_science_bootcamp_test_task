import pandas as pd


def dense_item_text(df: pd.DataFrame) -> pd.Series:
    """Текст объявления для bi-encoder: заголовок, начало параметров и описания (укладывается в 128 токенов)."""
    return (df.item_title_raw + " | " + df.item_infm_params_text.str.slice(0, 150) + " | "
            + df.item_description_raw.str.slice(0, 250)).str.strip(" |")


def dense_query_text(df: pd.DataFrame) -> pd.Series:
    """Текст запроса для bi-encoder: запрос плюс текстовые фильтры."""
    return (df.search_query + " " + df.search_infm_params_text).str.strip()


def bm25_item_text(df: pd.DataFrame) -> pd.Series:
    """Текст объявления для BM25: заголовок с весом 3, начало параметров и практически полное описание."""
    return ((df.item_title_raw + " . ") * 3 + df.item_infm_params_text.str.slice(0, 300) + " . "
            + df.item_description_raw.str.slice(0, 6000))
