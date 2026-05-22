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
STREAM_CSV       = PROJECT / "data/streaming/results.csv"
FILTER_CSV       = PROJECT / "data/streaming/filter_results.csv"
PREPROCESSED_CSV = PROJECT / "data/processed/preprocessed.csv"

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

# preprocessed_df: columnas OHE que el modelo KMeans espera.
# Solo se usa en enqueue_priority para publicar el registro correcto al topic de prioridad.
# Se carga con low_memory=False y se indexa por row_id para lookups O(1).
preprocessed_df = pd.read_csv(PREPROCESSED_CSV, low_memory=False)
preprocessed_df.insert(0, "row_id", np.arange(len(preprocessed_df), dtype=np.int32))
preprocessed_df = preprocessed_df.set_index("row_id")

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

ENRICH_COLS = ["row_id", "_name", "_country", "_city", "_price", "_rating",
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
    # Incluimos _name para que los registros SSE tengan el nombre legible
    enrich_cols = ENRICH_COLS + ["_name"]
    enriched = tmp.merge(df[enrich_cols], on="row_id", how="left")
    for col in ["_country", "_city", "_price", "_rating", "_name"]:
        enriched[col] = enriched[col].fillna("—")
    for col in ["vegetarian_friendly", "vegan_options", "gluten_free"]:
        enriched[col] = enriched[col].fillna(0).astype(int)
    enriched["_rating_float"] = pd.to_numeric(enriched["_rating"], errors="coerce").fillna(0.0)
    # Renombrar a los nombres que usa el frontend (sin prefijo _) para consistencia
    # con to_dict y to_dict_live, que son la fuente de datos del snapshot.
    enriched = enriched.rename(columns={
        "_name":    "name",
        "_country": "country",
        "_city":    "city",
        "_price":   "price",
        "_rating":  "rating",
    })
    # dist_to_centroid → dist para que coincida con to_dict / to_dict_live
    if "dist_to_centroid" in enriched.columns:
        enriched["dist"] = enriched["dist_to_centroid"].round(4)
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


# Caché de row_ids vistos por el consumidor de filtros (filter_results.csv).
# Solo contiene row_ids — el enriquecimiento se hace contra df en memoria.
_filter_ids_cache: set = set()
_filter_ids_mtime: float = -1.0

def get_filtered_row_ids() -> set:
    global _filter_ids_cache, _filter_ids_mtime
    if not FILTER_CSV.exists():
        return set()
    mtime = FILTER_CSV.stat().st_mtime
    if not _filter_ids_cache or mtime != _filter_ids_mtime:
        _filter_ids_cache = set(
            pd.read_csv(FILTER_CSV, usecols=["row_id"])["row_id"].astype(np.int32).tolist()
        )
        _filter_ids_mtime = mtime
    return _filter_ids_cache


def to_dict(row) -> dict:
    """Serializa una fila con cluster y dist garantizados (df estático o results.csv)."""
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


def to_dict_live(row) -> dict:
    """Serializa una fila del Live tab: cluster y dist pueden ser None
    para restaurantes vistos por el filtro pero aún no clasificados por KMeans.
    Acepta tanto los campos renombrados de enrich_stream (name, country, city...)
    como los campos con prefijo _ de df (para el modo búsqueda del snapshot)."""
    cluster = row.get("cluster")
    # dist puede venir como 'dist' (enrich_stream renombrado) o 'dist_to_centroid' (merge search mode)
    dist = row.get("dist") if row.get("dist") is not None else row.get("dist_to_centroid")
    return {
        "row_id":  int(row["row_id"]),
        "name":    row.get("name") or row.get("_name", "—"),
        "country": row.get("country") or row.get("_country", "—"),
        "city":    row.get("city") or row.get("_city", "—"),
        "rating":  row.get("rating") or row.get("_rating", "—"),
        "price":   row.get("price") or row.get("_price", "—"),
        "cluster": int(cluster) if pd.notna(cluster) else None,
        "dist":    round(float(dist), 4) if dist is not None and pd.notna(dist) else None,
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

    # Buscar el registro en preprocessed_df (columnas OHE que el modelo KMeans espera).
    # NO usar df (clean.csv) porque tiene columnas pre-encoding que producirían NaN
    # en el VectorAssembler y crashearían el scorer con una norma NaN.
    if row_id not in preprocessed_df.index:
        raise HTTPException(status_code=404, detail=f"row_id {row_id} no encontrado")

    record = preprocessed_df.loc[row_id].to_dict()
    record["row_id"] = int(row_id)
    # Los keys deben conservar los puntos (ej. avg_rating__1.0) para que Spark
    # los parsee correctamente contra spark_schema. streaming_score.py renombra
    # los puntos a '_' DESPUÉS del parseo, igual que hace con el stream normal.

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
    scored = get_enriched_stream()

    # Detectar si hay filtros activos para decidir el modo
    has_filters = any([
        country != "Todos", city != "Todos", price != "Todos",
        min_rating > 0.0, veg, vegan, gf,
    ])

    if not has_filters:
        # --- Modo overview: solo resultados KMeans, ordenados por llegada ---
        if scored.empty:
            return []
        return [to_dict_live(r) for r in (scored
                .sort_values("scored_at", ascending=False)
                .head(limit)
                .to_dict("records"))]

    # --- Modo búsqueda: combinar scored (results.csv) + unscored (filter_results.csv) ---
    # Partimos de los row_ids que el consumidor de filtros ha visto (más adelantado)
    filter_ids = get_filtered_row_ids()
    if not filter_ids:
        return []

    # Construimos un DataFrame base desde df para todos los row_ids del filtro,
    # con los campos de clean.csv correctamente decodificados
    base = df[df["row_id"].isin(filter_ids)][
        ["row_id", "_name", "_country", "_city", "_rating", "_rating_float",
         "_price", "vegetarian_friendly", "vegan_options", "gluten_free"]
    ].copy()

    # Aplicar filtros sobre el conjunto combinado
    if country != "Todos":  base = base[base["_country"] == country]
    if city != "Todos":     base = base[base["_city"] == city]
    if price != "Todos":    base = base[base["_price"] == price]
    if veg:                 base = base[base["vegetarian_friendly"] == 1]
    if vegan:               base = base[base["vegan_options"] == 1]
    if gf:                  base = base[base["gluten_free"] == 1]
    if min_rating > 0.0:    base = base[base["_rating_float"] >= min_rating]

    if base.empty:
        return []

    # Enriquecer con cluster y dist_to_centroid para los que ya han sido clasificados.
    # Los que no están en scored quedan con NaN en esas columnas → None en to_dict_live.
    # Forzamos int32 en ambos lados para evitar que el merge falle en silencio por
    # un mismatch de dtypes (df usa int32, results.csv se lee como int64 por defecto).
    if not scored.empty:
        scored_cols = scored[["row_id", "cluster", "dist", "scored_at"]].copy()
        scored_cols["row_id"] = scored_cols["row_id"].astype(np.int32)
        base["row_id"] = base["row_id"].astype(np.int32)
        base = base.merge(scored_cols, on="row_id", how="left")
    else:
        base["cluster"]          = None
        base["dist"] = None
        base["scored_at"]        = None

    # Ordenar: primero los ya clasificados (tienen scored_at), luego el resto
    base["_is_scored"] = base["scored_at"].notna()
    base = (base
            .sort_values(["_is_scored", "scored_at"], ascending=[False, False])
            .head(limit)
            .drop(columns=["_is_scored", "_rating_float"], errors="ignore"))

    return [to_dict_live(r) for r in base.to_dict("records")]


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
                        live_records = [to_dict_live(r) for r in enriched.to_dict("records")]
                        yield f"data: {json.dumps(live_records)}\n\n"
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
