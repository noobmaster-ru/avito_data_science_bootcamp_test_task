import re
import Stemmer

_stemmer = Stemmer.Stemmer("russian")
_cache: dict[str, str] = {}
_token_re = re.compile(r"[a-zа-я0-9]+")


def norm(s) -> str:
    """Нижний регистр, ё→е, только буквы/цифры, одиночные пробелы."""
    return " ".join(_token_re.findall(str(s).lower().replace("ё", "е")))


def stem_tokens(s) -> list[str]:
    """Стеммированные токены (Snowball ru) с кэшем по уникальным словам."""
    out = []
    for t in _token_re.findall(str(s).lower().replace("ё", "е")):
        r = _cache.get(t)
        if r is None:
            r = _cache[t] = _stemmer.stemWord(t)
        out.append(r)
    return out


def stem_text(s) -> str:
    """Стеммированный текст одной строкой (ключ для памяти кликов)."""
    return " ".join(stem_tokens(s))
