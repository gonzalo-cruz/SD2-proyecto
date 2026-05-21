import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

PROJECT = Path(__file__).parent.parent.parent
CLEAN_CSV = PROJECT / "data/processed/clean.csv"
ASSIGN_PQ = PROJECT / "models/cluster_assignments.parquet"
ENC_JSON = PROJECT / "data/processed/encodings.json"
STREAM_CSV = PROJECT / "data/streaming/results.csv"

app = FastAPI(title="RestaurantFindr API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargamos los datos de referencia al arrancar para tenerlos en memoria y acelerar las consultas.

enc = json.loads(ENC_JSON.read_text())

def make_decode(col):
    return {int(v): k for k, v in enc[col].items()} if col in enc else {}

decode_name = make_decode("restaurant_name")
decode_country = make_decode("country")
decode_city = make_decode("city")
decode_price = make_decode("price_level")
decode_rating = make_decode("avg_rating")

COLS = ["restaurant_name", "country", "city", "avg_rating",
        "price_level", "vegetarian_friendly", "vegan_options", "gluten_free"]

raw = pd.read_csv(CLEAN_CSV, usecols=COLS)
raw.insert(0, "row_id", np.arange(len(raw), dtype=np.int32))  # clean.csv no tiene IDs; índice posicional

assign = pd.read_parquet(ASSIGN_PQ)
df     = raw.merge(assign, on="row_id")

df["_name"] = df["restaurant_name"].map(decode_name).fillna("—")
df["_country"] = df["country"].map(decode_country).fillna("—")
df["_city"] = df["city"].map(decode_city).fillna("—")
df["_price"] = df["price_level"].map(decode_price).fillna("—")
df["_rating"] = (df["avg_rating"].map(decode_rating).fillna(df["avg_rating"])
                       if decode_rating else df["avg_rating"])
df["_rating_float"] = pd.to_numeric(df["_rating"], errors="coerce").fillna(0.0)

countries = sorted(df["_country"].dropna().unique().tolist())
prices = sorted(df["_price"][df["_price"] != "—"].unique().tolist())

# Matriz de similitud coseno (mismos features que KMeans) para recomendaciones rápidas dentro del cluster
# Se precalcula al arrancar para que las consultas de recomendaciones sean O(cluster_size),
# no O(N). Usar la misma métrica que el clustering mantiene coherencia en la definición
# de "similar" a lo largo de todo el sistema.

FEAT_COLS = ["country", "city", "avg_rating", "price_level",
             "vegetarian_friendly", "vegan_options", "gluten_free"]

_feat  = df[FEAT_COLS].values.astype(np.float32)
_norms = np.linalg.norm(_feat, axis=1, keepdims=True)
_norms[_norms == 0] = 1.0 # evitar división por cero en filas todo-cero
feat_norm = _feat / _norms # vectores unitarios; similitud coseno = producto punto
del _feat, _norms

ENRICH_COLS = ["row_id", "_country", "_city", "_price", "_rating",
               "vegetarian_friendly", "vegan_options", "gluten_free"]


def enrich_stream(raw_stream: pd.DataFrame) -> pd.DataFrame:
    tmp = raw_stream.copy()
    tmp["row_id"] = tmp["row_id"].astype(np.int32)  # el CSV lo lee como int64; df usa int32 → el merge fallaría en silencio
    enriched = tmp.merge(df[ENRICH_COLS], on="row_id", how="left")
    for col in ["_country", "_city", "_price", "_rating"]:
        enriched[col] = enriched[col].fillna("—")
    for col in ["vegetarian_friendly", "vegan_options", "gluten_free"]:
        enriched[col] = enriched[col].fillna(0).astype(int)
    enriched["_rating_float"] = pd.to_numeric(enriched["_rating"], errors="coerce").fillna(0.0)
    return enriched


# Caché del stream enriquecido: releer 800k filas en cada petición filtrada sería demasiado lento.
# Se invalida solo cuando el CSV cambia (mtime), lo que ocurre al llegar nuevos batches.
_stream_cache: pd.DataFrame = pd.DataFrame()
_stream_cache_mtime: float = 0.0

def get_enriched_stream() -> pd.DataFrame:
    global _stream_cache, _stream_cache_mtime
    if not STREAM_CSV.exists():
        return pd.DataFrame()
    mtime = STREAM_CSV.stat().st_mtime
    if _stream_cache.empty or mtime != _stream_cache_mtime:
        _stream_cache = enrich_stream(pd.read_csv(STREAM_CSV))
        _stream_cache_mtime = mtime
    return _stream_cache



def to_dict(row):
    return {
        "row_id": int(row["row_id"]),
        "name": row["_name"],
        "country": row["_country"],
        "city": row["_city"],
        "rating": row["_rating"],
        "price": row["_price"],
        "cluster": int(row["cluster"]),
        "dist": round(float(row["dist_to_centroid"]), 4),
        "veg": bool(row.get("vegetarian_friendly", 0)),
        "vegan": bool(row.get("vegan_options", 0)),
        "gf": bool(row.get("gluten_free", 0)),
    }


# Endpoints de la API

@app.get("/api/countries")
def get_countries():
    return countries


@app.get("/api/prices")
def get_prices():
    return prices


@app.get("/api/cities")
def get_cities(country: str = "Todos"):
    if country == "Todos":
        return []
    cities = df[df["_country"] == country]["_city"].dropna().unique().tolist()
    return sorted([c for c in cities if c != "—"])


@app.get("/api/restaurants")
def get_restaurants(
    country: str = "Todos",
    city: str = "Todos",
    min_rating: float = 0.0,
    price: str = "Todos",
    veg: bool = False,
    vegan: bool = False,
    gf: bool = False,
    search: str = "",
    limit: int = 100,
):
    filt = df
    if country != "Todos":
        filt = filt[filt["_country"] == country]
    if city != "Todos":
        filt = filt[filt["_city"] == city]
    if search:
        filt = filt[filt["_name"].str.contains(search, case=False, na=False)]
    if veg:
        filt = filt[filt["vegetarian_friendly"] == 1]
    if vegan:
        filt = filt[filt["vegan_options"] == 1]
    if gf:
        filt = filt[filt["gluten_free"] == 1]
    if min_rating > 0.0:
        filt = filt[filt["_rating_float"] >= min_rating]
    if price != "Todos":
        filt = filt[filt["_price"] == price]
    filt = filt.sort_values("dist_to_centroid").head(limit)
    return [to_dict(r) for r in filt.to_dict("records")]


@app.get("/api/recommendations/{row_id}")
def get_recommendations(row_id: int):
    row = df[df["row_id"] == row_id]
    if row.empty:
        return []
    idx = int(row.index[0])
    cluster = int(row.iloc[0]["cluster"])

    mask = (df["cluster"] == cluster) & (df["row_id"] != row_id)
    cluster_idx = df.index[mask].to_numpy()

    # Similitud coseno: producto punto entre vectores L2-normalizados.
    # Misma métrica que KMeans → definición de "similar" coherente con el clustering.
    sims  = feat_norm[cluster_idx] @ feat_norm[idx]
    top10 = np.argsort(sims)[::-1][:10]

    # Los cosenos dentro de un cluster son todos altos (0.99x), así que la similitud
    # absoluta sería 100% para todos. Se normaliza sobre el rango de los propios top-10
    # para que las diferencias sean visibles en la UI.
    top_sims = sims[top10]
    sim_max = float(top_sims[0])
    sim_min = float(top_sims[-1])
    sim_range = sim_max - sim_min

    result = []
    for rank, pos in enumerate(top10):
        d = to_dict(df.iloc[cluster_idx[pos]])
        if sim_range > 1e-6:
            d["similarity"] = round((float(sims[pos]) - sim_min) / sim_range * 100)
        else:
            # Todos igualmente similares (features idénticos en el cluster)
            d["similarity"] = max(0, 100 - rank * 10)
        result.append(d)
    return result


@app.get("/api/stream/snapshot")
def get_snapshot(
    limit: int = 500,
    country: str = "Todos",
    city: str = "Todos",
    price: str = "Todos",
    min_rating: float = 0.0,
    veg: bool = False,
    vegan: bool = False,
    gf: bool = False,
):
    # El filtrado es servidor porque con 800k filas el cliente solo recibiría
    # los últimos N cronológicos, que pueden no incluir el país/ciudad buscado.
    enriched = get_enriched_stream()
    if enriched.empty:
        return []
    if country != "Todos": enriched = enriched[enriched["_country"] == country]
    if city != "Todos": enriched = enriched[enriched["_city"] == city]
    if price != "Todos": enriched = enriched[enriched["_price"] == price]
    if veg: enriched = enriched[enriched["vegetarian_friendly"] == 1]
    if vegan: enriched = enriched[enriched["vegan_options"] == 1]
    if gf: enriched = enriched[enriched["gluten_free"] == 1]
    if min_rating > 0.0:
        enriched = enriched[enriched["_rating_float"] >= min_rating]
    return (enriched
            .sort_values("scored_at", ascending=False)
            .head(limit)
            .drop(columns=["_rating_float"], errors="ignore")
            .to_dict("records"))


@app.get("/api/stream/stats")
def get_stream_stats():
    if not STREAM_CSV.exists():
        return {"total": 0, "by_cluster": []}
    raw_stream = pd.read_csv(STREAM_CSV)
    counts = raw_stream["cluster"].value_counts()
    by_cluster = sorted(
        [{"cluster": int(k), "n": int(v)} for k, v in counts.items()],
        key=lambda x: x["cluster"],
    )
    return {"total": len(raw_stream), "by_cluster": by_cluster}


@app.get("/api/stream/events")
async def stream_events():
    async def generator():
        # Arrancar cerca del final del CSV para no reenviar el histórico completo;
        # el snapshot ya cubre los últimos N registros al montar el componente.
        try:
            snap_init = pd.read_csv(STREAM_CSV) if STREAM_CSV.exists() else pd.DataFrame()
            last = max(0, len(snap_init) - 50)
        except Exception:
            last = 0

        while True:
            try:
                if STREAM_CSV.exists():
                    snap = pd.read_csv(STREAM_CSV)
                    if len(snap) > last:
                        new_rows = snap.iloc[last:]
                        enriched = enrich_stream(new_rows)
                        last = len(snap)
                        yield f"data: {json.dumps(enriched.to_dict('records'))}\n\n"
                    else:
                        yield ": ping\n\n"   # keepalive para que el navegador no cierre la conexión
            except Exception:
                yield ": error\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
