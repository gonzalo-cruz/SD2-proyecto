import asyncio
import json
from pathlib import Path

import numpy as np
import orjson
import pandas as pd
from confluent_kafka import Producer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

PROJECT    = Path(__file__).parent.parent.parent
CLEAN_CSV  = PROJECT / "data/processed/clean.csv"
ASSIGN_PQ  = PROJECT / "models/cluster_assignments.parquet"
ENC_JSON   = PROJECT / "data/processed/encodings.json"
STREAM_CSV = PROJECT / "data/streaming/results.csv"
FILTER_CSV = PROJECT / "data/streaming/filter_results.csv"

# Topic de Kafka al que se publican los restaurantes que el usuario selecciona
# antes de que el consumidor KMeans haya llegado a ellos.
PRIORITY_TOPIC = "restaurants_priority"
KAFKA_SERVERS  = "localhost:9092"

app = FastAPI(title="RestaurantFindr API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Datos de referencia cargados al arrancar
# ---------------------------------------------------------------------------

enc = json.loads(ENC_JSON.read_text())

def make_decode(col):
    return {int(v): k for k, v in enc[col].items()} if col in enc else {}

decode_name    = make_decode("restaurant_name")
decode_country = make_decode("country")
decode_city    = make_decode("city")
decode_price   = make_decode("price_level")
decode_rating  = make_decode("avg_rating")

COLS = ["restaurant_name", "country", "city", "avg_rating",
        "price_level", "vegetarian_friendly", "vegan_options", "gluten_free"]

raw = pd.read_csv(CLEAN_CSV, usecols=COLS)
raw.insert(0, "row_id", np.arange(len(raw), dtype=np.int32))

assign = pd.read_parquet(ASSIGN_PQ)
df     = raw.merge(assign, on="row_id")

df["_name"]    = df["restaurant_name"].map(decode_name).fillna("—")
df["_country"] = df["country"].map(decode_country).fillna("—")
df["_city"]    = df["city"].map(decode_city).fillna("—")
df["_price"]   = df["price_level"].map(decode_price).fillna("—")
df["_rating"]  = (df["avg_rating"].map(decode_rating).fillna(df["avg_rating"])
                  if decode_rating else df["avg_rating"])
df["_rating_float"] = pd.to_numeric(df["_rating"], errors="coerce").fillna(0.0)

countries = sorted(df["_country"].dropna().unique().tolist())
prices    = sorted(df["_price"][df["_price"] != "—"].unique().tolist())

# ---------------------------------------------------------------------------
# Matriz de similitud coseno precomputada (para recomendaciones)
# ---------------------------------------------------------------------------

FEAT_COLS = ["country", "city", "avg_rating", "price_level",
             "vegetarian_friendly", "vegan_options", "gluten_free"]

_feat  = df[FEAT_COLS].values.astype(np.float32)
_norms = np.linalg.norm(_feat, axis=1, keepdims=True)
_norms[_norms == 0] = 1.0
feat_norm = _feat / _norms
del _feat, _norms

ENRICH_COLS = ["row_id", "_country", "_city", "_price", "_rating",
               "vegetarian_friendly", "vegan_options", "gluten_free"]

# ---------------------------------------------------------------------------
# Productor Kafka para la cola de prioridad
# ---------------------------------------------------------------------------

_kafka_producer: Producer | None = None

def get_kafka_producer() -> Producer:
    global _kafka_producer
    if _kafka_producer is None:
        _kafka_producer = Producer({"bootstrap.servers": KAFKA_SERVERS, "acks": "all"})
    return _kafka_producer

# ---------------------------------------------------------------------------
# Helpers de enriquecimiento y caché del stream
# ---------------------------------------------------------------------------

def enrich_stream(raw_stream: pd.DataFrame) -> pd.DataFrame:
    tmp = raw_stream.copy()
    tmp["row_id"] = tmp["row_id"].astype(np.int32)
    enriched = tmp.merge(df[ENRICH_COLS], on="row_id", how="left")
    for col in ["_country", "_city", "_price", "_rating"]:
        enriched[col] = enriched[col].fillna("—")
    for col in ["vegetarian_friendly", "vegan_options", "gluten_free"]:
        enriched[col] = enriched[col].fillna(0).astype(int)
    enriched["_rating_float"] = pd.to_numeric(enriched["_rating"], errors="coerce").fillna(0.0)
    return enriched


def enrich_filter(raw_filter: pd.DataFrame) -> pd.DataFrame:
    tmp = raw_filter.copy()
    tmp["row_id"] = tmp["row_id"].astype(np.int32)
    # Los mensajes Kafka tienen datos OHE (preprocessed.csv), no los campos simples.
    # Enriquecemos mergeando con df (clean.csv) por row_id, igual que enrich_stream.
    enriched = tmp.merge(df[ENRICH_COLS + ["_name"]], on="row_id", how="left")
    for col in ["_country", "_city", "_price", "_rating", "_name"]:
        enriched[col] = enriched[col].fillna("—")
    for col in ["vegetarian_friendly", "vegan_options", "gluten_free"]:
        enriched[col] = enriched[col].fillna(0).astype(int)
    enriched["_rating_float"] = pd.to_numeric(enriched["_rating"], errors="coerce").fillna(0.0)
    return enriched


# Caché del stream KMeans (resultados clasificados)
_stream_cache: pd.DataFrame = pd.DataFrame()
_stream_cache_mtime: float = -1.0

def get_enriched_stream() -> pd.DataFrame:
    global _stream_cache, _stream_cache_mtime
    if not STREAM_CSV.exists():
        return pd.DataFrame()
    mtime = STREAM_CSV.stat().st_mtime
    if _stream_cache.empty or mtime != _stream_cache_mtime:
        _stream_cache = enrich_stream(pd.read_csv(STREAM_CSV))
        _stream_cache_mtime = mtime
    return _stream_cache


# Caché del stream de filtros (resultados sin clasificar, llegan antes)
_filter_cache: pd.DataFrame = pd.DataFrame()
_filter_cache_mtime: float = -1.0

def get_enriched_filter() -> pd.DataFrame:
    global _filter_cache, _filter_cache_mtime
    if not FILTER_CSV.exists():
        return pd.DataFrame()
    mtime = FILTER_CSV.stat().st_mtime
    if _filter_cache.empty or mtime != _filter_cache_mtime:
        _filter_cache = enrich_filter(pd.read_csv(FILTER_CSV))
        _filter_cache_mtime = mtime
    return _filter_cache


def to_dict_filter(row) -> dict:
    """Serializa una fila del consumidor de filtros (sin cluster ni dist)."""
    # cluster y dist pueden no existir todavía; los marcamos como None
    scored = get_enriched_stream()
    cluster = None
    dist    = None
    if not scored.empty:
        match = scored[scored["row_id"] == int(row["row_id"])]
        if not match.empty:
            cluster = int(match.iloc[0]["cluster"])
            dist    = round(float(match.iloc[0]["dist_to_centroid"]), 4)

    return {
        "row_id":   int(row["row_id"]),
        "name":     row["_name"],
        "country":  row["_country"],
        "city":     row["_city"],
        "rating":   row["_rating"],
        "price":    row["_price"],
        "cluster":  cluster,   # None si aún no ha pasado por KMeans
        "dist":     dist,      # None si aún no ha pasado por KMeans
        "veg":      bool(row.get("vegetarian_friendly", 0)),
        "vegan":    bool(row.get("vegan_options", 0)),
        "gf":       bool(row.get("gluten_free", 0)),
    }


def to_dict(row) -> dict:
    """Serializa una fila del consumidor KMeans (con cluster y dist)."""
    return {
        "row_id":  int(row["row_id"]),
        "name":    row["_name"],
        "country": row["_country"],
        "city":    row["_city"],
        "rating":  row["_rating"],
        "price":   row["_price"],
        "cluster": int(row["cluster"]),
        "dist":    round(float(row["dist_to_centroid"]), 4),
        "veg":     bool(row.get("vegetarian_friendly", 0)),
        "vegan":   bool(row.get("vegan_options", 0)),
        "gf":      bool(row.get("gluten_free", 0)),
    }

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

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
    """
    MODIFICADO: ahora lee de filter_results.csv (consumidor rápido sin ML)
    en lugar de results.csv (consumidor KMeans).
    Esto garantiza que el usuario vea resultados incluso cuando el consumidor
    KMeans todavía no ha procesado esos registros.
    Los campos 'cluster' y 'dist' pueden ser None para restaurantes aún no
    clasificados por KMeans.
    """
    filt = get_enriched_filter()
    if filt.empty:
        return []

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

    filt = filt.head(limit)
    return [to_dict_filter(r) for r in filt.to_dict("records")]


@app.post("/api/priority/{row_id}", status_code=202)
def enqueue_priority(row_id: int):
    """
    NUEVO: Publica el restaurante indicado en el topic de prioridad de Kafka
    para que el consumidor KMeans lo clasifique inmediatamente, sin esperar
    a que el stream normal llegue a ese registro.

    Se llama desde el frontend cuando el usuario selecciona un restaurante
    que aún no tiene cluster asignado (cluster == None en /api/restaurants).

    Pasos:
      1. Verificar que el restaurante no está ya clasificado (evitar duplicados).
      2. Recuperar el registro crudo del dataset limpio.
      3. Publicarlo en 'restaurants_priority'.
    """
    # Comprobar si ya está clasificado — si es así no hace falta encolar
    scored = get_enriched_stream()
    if not scored.empty and not scored[scored["row_id"] == row_id].empty:
        return {"status": "already_scored", "row_id": row_id}

    # Buscar el registro en el dataset de referencia
    row = df[df["row_id"] == row_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"row_id {row_id} no encontrado")

    # Construir el mensaje con los campos que el consumidor KMeans espera
    record = row.iloc[0].to_dict()
    record["row_id"] = int(row_id)

    # Serializar y publicar en Kafka
    try:
        producer = get_kafka_producer()
        producer.produce(
            topic=PRIORITY_TOPIC,
            value=orjson.dumps(record),
        )
        producer.poll(0)
        producer.flush(timeout=2.0)
    except Exception as exc:
        raise HTTPException(status_code=503,
                            detail=f"Error publicando en Kafka: {exc}")

    return {"status": "queued", "row_id": row_id}


@app.get("/api/recommendations/{row_id}")
def get_recommendations(row_id: int):
    """
    MODIFICADO: si el restaurante aún no ha sido clasificado por KMeans,
    devuelve {"status": "pending"} en lugar de una lista vacía, para que
    el frontend sepa que debe reintentar después de encolar una solicitud
    de prioridad (POST /api/priority/{row_id}).
    """
    row = df[df["row_id"] == row_id]
    if row.empty:
        return []

    # Intentar obtener el cluster del stream KMeans
    scored = get_enriched_stream()
    if scored.empty or scored[scored["row_id"] == row_id].empty:
        # Aún no clasificado
        return {"status": "pending", "row_id": row_id}

    idx     = int(row.index[0])
    cluster = int(row.iloc[0]["cluster"])

    mask        = (df["cluster"] == cluster) & (df["row_id"] != row_id)
    cluster_idx = df.index[mask].to_numpy()

    sims  = feat_norm[cluster_idx] @ feat_norm[idx]
    top10 = np.argsort(sims)[::-1][:10]

    top_sims  = sims[top10]
    sim_max   = float(top_sims[0])
    sim_min   = float(top_sims[-1])
    sim_range = sim_max - sim_min

    result = []
    for rank, pos in enumerate(top10):
        d = to_dict(df.iloc[cluster_idx[pos]])
        if sim_range > 1e-6:
            d["similarity"] = round((float(sims[pos]) - sim_min) / sim_range * 100)
        else:
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
    # El snapshot sigue usando el stream KMeans (tiene cluster y dist)
    enriched = get_enriched_stream()
    if enriched.empty:
        return []
    if country != "Todos":   enriched = enriched[enriched["_country"] == country]
    if city != "Todos":      enriched = enriched[enriched["_city"] == city]
    if price != "Todos":     enriched = enriched[enriched["_price"] == price]
    if veg:                  enriched = enriched[enriched["vegetarian_friendly"] == 1]
    if vegan:                enriched = enriched[enriched["vegan_options"] == 1]
    if gf:                   enriched = enriched[enriched["gluten_free"] == 1]
    if min_rating > 0.0:     enriched = enriched[enriched["_rating_float"] >= min_rating]
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


@app.get("/api/filter/stats")
def get_filter_stats():
    if not FILTER_CSV.exists():
        return {"total": 0}
    with open(FILTER_CSV) as f:
        total = sum(1 for _ in f) - 1
    return {"total": max(0, total)}


@app.get("/api/stream/events")
async def stream_events():
    async def generator():
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
                        yield ": ping\n\n"
            except Exception:
                yield ": error\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
