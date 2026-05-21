# Carga y preprocesado en memoria de los datos para la aplicación.
# El decorador @st.cache_data hace que Streamlit solo ejecute esto una vez
# (la primera carga); las siguientes llamadas devuelven el resultado cacheado.
import json

import polars as pl
import streamlit as st

from config import ASSIGNMENTS_PQ, CLEAN_CSV, CUISINES_JSON, ENCODINGS_JSON


@st.cache_data
def load_data():
    # Si el modelo no ha sido entrenado todavía no hay assignments
    if not ASSIGNMENTS_PQ.exists():
        return None, None, None, None, None, None

    # Leemos el CSV limpio y añadimos un índice de fila para poder
    # cruzarlo con los resultados del clustering y del stream
    df = (pl.read_csv(CLEAN_CSV)
            .with_row_index("row_id")
            .with_columns(pl.col("row_id").cast(pl.Int32)))

    # Unimos los clusters asignados por KMeans (generados en train_model.py)
    assignments = pl.read_parquet(ASSIGNMENTS_PQ)
    df = df.join(assignments, on="row_id", how="left")

    with open(ENCODINGS_JSON) as f:
        enc = json.load(f)

    # decode: código numérico → etiqueta legible (p.ej. 3 → "España")
    decode = {}
    for col in ("country", "city", "restaurant_name",
                "avg_rating", "price_level",
                "vegetarian_friendly", "vegan_options", "gluten_free"):
        if col in enc:
            decode[col] = {int(v): k for k, v in enc[col].items()}

    # encode: etiqueta → código numérico, necesario para filtrar con Polars
    encode = {
        "country": {k: int(v) for k, v in enc.get("country", {}).items()},
        "city": {k: int(v) for k, v in enc.get("city",    {}).items()},
        "price_level": {k: int(v) for k, v in enc.get("price_level", {}).items()},
        "avg_rating": {k: int(v) for k, v in enc.get("avg_rating",  {}).items()},
    }

    # Construimos un mapa país → lista de ciudades para el filtro en cascada
    country_cities: dict[int, list[str]] = {}
    for country_code, city_code in df.select(["country", "city"]).unique().iter_rows():
        name = decode["city"].get(int(city_code), str(city_code))
        country_cities.setdefault(int(country_code), []).append(name)
    for cc in country_cities:
        country_cities[cc] = sorted(set(country_cities[cc]))

    # cuisine_idx: tipo de cocina → conjunto de row_ids que la tienen
    # (las cocinas están en un fichero separado porque un restaurante puede tener varias)
    with open(CUISINES_JSON) as f:
        raw = json.load(f)

    cuisine_idx: dict[str, set] = {}
    for entry in raw:
        cuisine_idx.setdefault(entry["value"], set()).add(entry["row_id"])
    # Mostramos solo las 20 cocinas más frecuentes en el filtro
    top_cuisines = sorted(cuisine_idx, key=lambda c: -len(cuisine_idx[c]))[:20]

    # Añadimos una columna con el nombre decodificado para la búsqueda por texto,
    # calculada una sola vez aquí en lugar de hacerlo en cada interacción
    code_to_name = decode["restaurant_name"]
    name_list = [code_to_name.get(int(c), "") for c in df["restaurant_name"].to_list()]
    df = df.with_columns(pl.Series("_name_str", name_list, dtype=pl.Utf8))

    return df, decode, encode, cuisine_idx, top_cuisines, country_cities
