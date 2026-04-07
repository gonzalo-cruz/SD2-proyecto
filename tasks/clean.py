import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Rutas
RAW_CSV = Path(__file__).parent.parent / "data" / "raw" / "raw.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"

CHUNK_SIZE = 50_000
NULL_THRESHOLD = 0.70  # columnas con más del 70% de nulos se eliminan

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


NUMERIC_CATEGORICAL_THRESHOLD = 20  # columnas numéricas con <= 20 valores únicos → numeric_categorical
NUMERIC_DISCRETE_THRESHOLD = 200    # columnas enteras con <= 200 valores únicos → numeric_discrete


def classify_numeric(series):
    """
    Subclasifica una serie numérica en:
      - numeric_categorical -> pocos valores únicos, probablemente una categoría codificada
      - numeric_discrete    -> valores enteros con cardinalidad moderada
      - numeric_continuous  -> valores continuos (float o muchos enteros únicos)
    """
    n_unique = series.nunique()

    if n_unique <= NUMERIC_CATEGORICAL_THRESHOLD:
        return "numeric_categorical"

    is_integer_valued = (series.dropna() % 1 == 0).all()
    if is_integer_valued and n_unique <= NUMERIC_DISCRETE_THRESHOLD:
        return "numeric_discrete"

    return "numeric_continuous"


def detect_column_type(series):
    """
    Clasifica una columna en uno de estos tipos:
      - numeric_continuous  -> números continuos (float o muchos valores únicos)
      - numeric_discrete    -> enteros con cardinalidad moderada
      - numeric_categorical -> numérica con pocos valores únicos (probablemente categoría)
      - boolean    -> solo valores Y / N
      - list_json  -> listas o dicts en formato JSON (empieza con [ o {)
      - list_csv   -> listas separadas por comas (ej: "Lunch, Dinner")
      - categorical -> texto plano
    """
    series = series.dropna()

    # ¿Es numérica?
    if pd.api.types.is_numeric_dtype(series):
        return classify_numeric(series)

    # ¿Se puede convertir a número?
    try:
        numeric_series = pd.to_numeric(series)
        return classify_numeric(numeric_series)
    except (ValueError, TypeError):
        pass

    sample = series.astype(str)

    # ¿Solo Y/N?
    if set(sample.str.strip().unique()) <= {"Y", "N"}:
        return "boolean"

    # ¿La mayoría empieza con [ o {? → lista JSON
    starts_with_bracket = sample.str.strip().str[:1].isin(["[", "{"]).mean()
    if starts_with_bracket > 0.5:
        return "list_json"

    # ¿La mayoría tiene comas? → lista CSV
    has_comma = sample.str.contains(",", regex=False).mean()
    if has_comma > 0.3:
        return "list_csv"

    return "categorical"

def clean():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # calcular el % de nulos por columna recorriendo el CSV
    log.info(" Calculando porcentaje de nulos por columna...")

    null_counts = None
    total_rows = 0
    all_columns = []

    for chunk in pd.read_csv(RAW_CSV, chunksize=CHUNK_SIZE, low_memory=False):
        if null_counts is None:
            null_counts = chunk.isnull().sum()
            all_columns = list(chunk.columns)
        else:
            null_counts += chunk.isnull().sum()
        total_rows += len(chunk)

    null_ratio = null_counts / total_rows

    # Columnas a eliminar (más del 70% de nulos)
    drop_columns = list(null_ratio[null_ratio > NULL_THRESHOLD].index)
    keep_columns = [c for c in all_columns if c not in drop_columns]

    log.info("Columnas eliminadas (>%.0f%% nulos): %s", NULL_THRESHOLD * 100, drop_columns)
    log.info("Columnas que se mantienen: %d / %d", len(keep_columns), len(all_columns))

    # Detectar el tipo de cada columna con una muestra pequeña
    log.info("Detectando tipos de columnas...")

    sample = pd.read_csv(RAW_CSV, usecols=keep_columns, nrows=10_000, low_memory=False)
    type_dict = {}
    for col in keep_columns:
        type_dict[col] = detect_column_type(sample[col])

    # Resumen de tipos
    from collections import Counter
    log.info("Tipos detectados: %s", dict(Counter(type_dict.values())))

    # Calcular valores de imputación y encoding recorriendo el CSV
    log.info("Paso 3: calculando estadísticas para imputación y encoding...")

    NUMERIC_TYPES = {"numeric_continuous", "numeric_discrete"}
    CATEGORICAL_LIKE = {"categorical", "boolean", "numeric_categorical"}

    # Acumulamos valores numéricos y conteos de categorías
    numeric_values = {col: [] for col, t in type_dict.items() if t in NUMERIC_TYPES}
    category_counts = {col: {} for col, t in type_dict.items() if t in CATEGORICAL_LIKE}

    for i, chunk in enumerate(pd.read_csv(RAW_CSV, usecols=keep_columns, chunksize=CHUNK_SIZE, low_memory=False)):
        for col in numeric_values:
            numeric_values[col].extend(chunk[col].dropna().tolist())

        for col in category_counts:
            for val in chunk[col].dropna().astype(str):
                category_counts[col][val] = category_counts[col].get(val, 0) + 1

        log.info("  stats batch %d completado", i + 1)

    # Valor de imputación por columna
    fill_values = {}
    for col, dtype in type_dict.items():
        if dtype in NUMERIC_TYPES:
            vals = numeric_values.get(col, [])
            fill_values[col] = float(np.median(vals)) if vals else 0.0
        elif dtype in CATEGORICAL_LIKE:
            counts = category_counts.get(col, {})
            fill_values[col] = max(counts, key=counts.get) if counts else ""
        else:
            fill_values[col] = ""  # listas → cadena vacía

    # Encoding -> asignamos un entero a cada valor único de columnas categóricas/booleanas y numeric_categorical
    encodings = {}
    for col, counts in category_counts.items():
        sorted_values = sorted(counts.keys())
        encodings[col] = {val: idx for idx, val in enumerate(sorted_values)}

    # PASO 4: transformar el dataset en batches y guardar
    log.info("Transformando datos y guardando clean.csv...")

    output_csv = OUTPUT_DIR / "clean.csv"
    total_rows_written = 0
    first_chunk = True

    for i, chunk in enumerate(pd.read_csv(RAW_CSV, usecols=keep_columns, chunksize=CHUNK_SIZE, low_memory=False)):

        # Imputar nulos
        for col, dtype in type_dict.items():
            if col in chunk.columns and chunk[col].isna().any():
                chunk[col] = chunk[col].fillna(fill_values[col])

        # Convertir columnas numéricas que quedaron como string
        for col, dtype in type_dict.items():
            if dtype in NUMERIC_TYPES and col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(fill_values[col])

        # Label encoding-> reemplazar strings por su código entero
        for col, mapping in encodings.items():
            if col in chunk.columns:
                chunk[col] = chunk[col].astype(str).map(mapping)

        chunk.to_csv(output_csv, mode="w" if first_chunk else "a", header=first_chunk, index=False)
        first_chunk = False
        total_rows_written += len(chunk)
        log.info("  transform batch %d — %d filas escritas", i + 1, total_rows_written)

    log.info("clean.csv guardado: %d filas", total_rows_written)

    # Guardar artefactos
    with open(OUTPUT_DIR / "type_dict.json", "w", encoding="utf-8") as f:
        json.dump(type_dict, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_DIR / "encodings.json", "w", encoding="utf-8") as f:
        json.dump(encodings, f, indent=2, ensure_ascii=False)

    log.info("Artefactos guardados: type_dict.json, encodings.json")


if __name__ == "__main__":
    clean()
