import pandas as pd, numpy as np, re
from cg.data import load
tr = load("train"); bi = load("benchmark_items"); bq = load("benchmark_queries")
tr_mc = set(tr.item_microcat_id.dropna())
print("share corpus items with microcat seen in train", round(bi.item_microcat_id.isin(tr_mc).mean(),4))
print("corpus microcats top", bi.item_microcat_id.value_counts().head(10).to_dict())
print("train microcats top", tr.item_microcat_id.value_counts().head(10).to_dict())
vid = re.compile(r"Вид услуги ([^А-ЯЁ]*?)(?= [А-ЯЁ]|$)")
def kinds(s): return set(m.strip() for m in vid.findall(s))
m = tr.search_infm_params_text != ""
sub = tr[m].sample(20000, random_state=0)
qk = sub.search_infm_params_text.map(kinds); ik = sub.item_infm_params_text.map(kinds)
has = qk.map(len) > 0
print("share of non-empty search params with 'Вид услуги'", round(has.mean(),3))
print("P(item kind ⊇ search kind | has)", round(np.mean([k <= i for k, i, h in zip(qk, ik, has) if h]),4))
print("examples mismatch:", [(k, i) for k, i, h in zip(qk, ik, has) if h and not k <= i][:3])
tip = re.compile(r"Тип услуги ([^А-ЯЁ]*?)(?= [А-ЯЁ]|$)")
def types(s): return set(m.strip() for m in tip.findall(s))
qt = sub.search_infm_params_text.map(types); it = sub.item_infm_params_text.map(types)
hast = qt.map(len) > 0
print("share with 'Тип услуги'", round(hast.mean(),3), "P(item type ⊇ search type | has)", round(np.mean([k <= i for k, i, h in zip(qt, it, hast) if h]),4))
print("examples type mismatch:", [(k, i) for k, i, h in zip(qt, it, hast) if h and not k <= i][:3])
print("bq params top:\n", bq.search_infm_params_text.value_counts().head(10).to_string())
# location mismatch: are item coords of mismatched near same-location centroid?
bi["lat"] = pd.to_numeric(bi.item_latitude, errors="coerce"); bi["lon"] = pd.to_numeric(bi.item_longitude, errors="coerce")
cent = bi.groupby("item_location_id")[["lat","lon"]].median()
tr["lat"] = pd.to_numeric(tr.item_latitude, errors="coerce"); tr["lon"] = pd.to_numeric(tr.item_longitude, errors="coerce")
s = tr.sample(50000, random_state=0)
c = cent.reindex(s.search_location_id.values)
d = np.sqrt(((s.lat.values - c.lat.values)*111)**2 + ((s.lon.values - c.lon.values)*111*np.cos(np.radians(55)))**2)
mm = s.item_location_id.values != s.search_location_id.values
print("centroid known for search loc share", round(np.isfinite(d).mean(),3))
print("dist km quantiles, same-loc", np.nanquantile(d[~mm], [.5,.9,.99]).round(1), "diff-loc", np.nanquantile(d[mm], [.5,.9,.99]).round(1))
print("counts in q text:", tr.q.value_counts().head(10).to_dict() if "q" in tr else "")
