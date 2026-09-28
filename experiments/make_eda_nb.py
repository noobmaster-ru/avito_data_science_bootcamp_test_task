"""Генерирует notebooks/01_eda.ipynb: разведочный анализ, определивший архитектуру решения."""
import nbformat as nbf
nb = nbf.v4.new_notebook(); c = nb.cells
md = lambda s: c.append(nbf.v4.new_markdown_cell(s)); code = lambda s: c.append(nbf.v4.new_code_cell(s))
md("""# 01. EDA: что в данных определяет архитектуру кандидатогенерации

Задача: по запросу отобрать до 50 объявлений из корпуса (189 212), метрика Recall@50.
Здесь проверяем гипотезы, от которых зависит выбор подхода:
1. насколько бенчмарк пересекается с train по объявлениям и по текстам запросов (сила «памяти кликов»);
2. годятся ли категория и локация как фильтры;
3. насколько однозначно текст запроса определяет микрокатегорию;
4. что лежит в текстовых полях и фильтрах.""")
code("""import sys; sys.path.insert(0, "../src")
import numpy as np, pandas as pd
from cg.data import load
from cg.text import norm, stem_text
tr, bq, bi = load("train"), load("benchmark_queries"), load("benchmark_items")
tr["q"], bq["q"] = tr.search_query.map(norm), bq.search_query.map(norm)
print(tr.shape, bq.shape, bi.shape)""")
md("## 1. Пересечение бенчмарка с train\n\nЕсли объявления бенчмарка встречались в train, работает «память»: запрос → объявления, которые по нему уже выбирали.")
code("""tr_items, bi_items = set(tr.item_id), set(bi.item_id)
print("доля корпуса, встречающаяся в train:", round(len(bi_items & tr_items) / len(bi_items), 3))
print("доля строк train с объявлением из корпуса:", round(tr.item_id.isin(bi_items).mean(), 3))
print("доля запросов бенчмарка с таким же текстом в train:", round(bq.q.isin(set(tr.q)).mean(), 3))
print("уникальных текстов среди запросов бенчмарка:", bq.q.nunique(), "из", len(bq))""")
md("""**Вывод.** Только 9,6 % корпуса встречалось в train, а тексты запросов бенчмарка уникальны и лишь на 37,5 % виделись в train.
Память кликов не может быть основой решения: нужен текстовый поиск по всему корпусу. Локальная валидация должна имитировать
бенчмарк: уникальные тексты запросов и та же доля «виденных» (см. `cg/validation.py`).""")
md("## 2. Категория и локация")
code("""print("P(категория объявления == категория поиска):", round((tr.item_category_id == tr.search_category).mean(), 4))
print("категории поиска в train:", tr.search_category.value_counts().head(3).to_dict())
print("категории поиска в бенчмарке:", bq.search_category.value_counts().head(3).to_dict())
print("P(локация объявления == локация поиска):", round((tr.item_location_id == tr.search_location_id).mean(), 3))""")
code("""bi["lat"], bi["lon"] = pd.to_numeric(bi.item_latitude, errors="coerce"), pd.to_numeric(bi.item_longitude, errors="coerce")
cent = bi.groupby("item_location_id")[["lat", "lon"]].median()
s = tr.sample(50_000, random_state=0)
s_lat, s_lon = pd.to_numeric(s.item_latitude, errors="coerce").values, pd.to_numeric(s.item_longitude, errors="coerce").values
c = cent.reindex(s.search_location_id.values)
d = np.sqrt(((s_lat - c.lat.values) * 111) ** 2 + ((s_lon - c.lon.values) * 111 * np.cos(np.radians(55))) ** 2)
mm = s.item_location_id.values != s.search_location_id.values
print("км до центроида локации поиска, та же локация:", np.nanquantile(d[~mm], [.5, .9, .99]).round(1))
print("км до центроида локации поиска, другая локация:", np.nanquantile(d[mm], [.5, .9, .99]).round(1))""")
md("""**Вывод.** Категория одна и та же почти всегда (услуги), как признак бесполезна. Локация совпадает в 83 % случаев,
а несовпадающие объявления обычно находятся недалеко (медиана 34 км). Поэтому локация используется не как жёсткий фильтр,
а как множитель к скору: бонус за совпадение `location_id` и бонус, убывающий с расстоянием до центроида локации запроса.""")
md("## 3. Текст запроса и микрокатегория")
code("""g = tr.groupby("q")
print("объявлений на текст запроса (квантили):", g.item_id.nunique().quantile([.5, .9, .99]).to_dict())
print("микрокатегорий на текст запроса (квантили):", g.item_microcat_id.nunique().quantile([.5, .9, .99]).to_dict())
print("микрокатегорий в train / в корпусе:", tr.item_microcat_id.nunique(), "/", bi.item_microcat_id.nunique())""")
md("""**Вывод.** У виденного текста запроса почти всегда одна микрокатегория, но в корпусе микрокатегорий втрое больше, чем в train,
а классификатор «текст → микрокатегория» на невиденных запросах даёт лишь ~53 % top-1. Как жёсткий буст он вредил на валидации,
поэтому вероятность микрокатегории используется только как признак реранкера.""")
md("## 4. Тексты и фильтры")
code("""for col in ["search_query", "item_title_raw", "item_description_raw", "item_infm_params_text"]:
    src = tr if col in tr else bi
    print(col, "длина (квантили):", src[col].str.len().quantile([.5, .9, .99]).to_dict())
print("доля пустых фильтров: train", round((tr.search_infm_params_text == "").mean(), 3), "бенчмарк", round((bq.search_infm_params_text == "").mean(), 3))
print(bq.search_infm_params_text.value_counts().head(6).to_string())
print(bi.item_infm_params_text.iloc[0][:300])""")
md("""**Вывод.** Запросы короткие (медиана 17 символов), заголовки информативные, параметры объявления длинные и шаблонные
(«Вид услуги …», «Услуга … Стоимость …»). В индексы идёт заголовок с повышенным весом, начало параметров и описания;
текст фильтров запроса добавляется к тексту запроса.""")
nbf.write(nb, "notebooks/01_eda.ipynb")
