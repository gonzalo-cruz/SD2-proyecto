import json
import logging
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA

# Rutas
INPUT_CSV = Path(__file__).parent.parent/"data"/"eda"/"clean_no_outliers.csv"
TYPE_DICT = Path(__file__).parent.parent/"data"/"processed"/"type_dict.json"
ENCODINGS = Path(__file__).parent.parent/"data"/"processed"/"encodings.json"
OUTPUT_DIR = Path(__file__).parent.parent/"data"/"processed"

CHUNK_SIZE = 50_000
OHE_MAX_CARDINALITY = 15  # columnas con <= 15 valores únicos se codifican con OHE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def preprocessing():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Cargamos el diccionario de los tipos y los encodings para saber que hacer con cada columna
    with open(TYPE_DICT, encoding="utf-8") as f:
        type_dict = json.load(f)
    with open(ENCODINGS, encoding="utf-8") as f:
        encodings = json.load(f)

    # Separamos columnas por cmo las vamos a tratar
    numeric_cols = [c for c, t in type_dict.items() if t == "numeric"]
    ohe_cols     = [c for c, t in type_dict.items()
                    if t == "encoded_categorical" and len(encodings.get(c, {})) <= OHE_MAX_CARDINALITY]
    label_cols   = [c for c, t in type_dict.items()
                    if t == "encoded_categorical" and len(encodings.get(c, {})) > OHE_MAX_CARDINALITY]
    passthrough  = [c for c, t in type_dict.items() if t in ("list_json", "list_csv")]

    log.info("Numericas: %d | OHE: %d | Label encoding: %d | Sin cambios: %d",
             len(numeric_cols), len(ohe_cols), len(label_cols), len(passthrough))

    # Preparamos el escalado y el pca
    log.info("Escalando numéricas y ajustando PCA...")

    scaler = StandardScaler()
    n_components = max(1, min(len(numeric_cols) - 1, 50))
    ipca = IncrementalPCA(n_components=n_components)

    for i, chunk in enumerate(pd.read_csv(INPUT_CSV, usecols=numeric_cols, chunksize=CHUNK_SIZE, low_memory=False)):
        X = chunk[numeric_cols].to_numpy(dtype=np.float64)

        # partial_fit actualiza las estadísticas del scaler sin transformar
        scaler.partial_fit(X)

        # Escalar antes de ajustar el PCA (PCA necesita datos ya escalados)
        X_scaled = scaler.transform(X)
        if len(X_scaled) >= n_components:
            ipca.partial_fit(X_scaled)

        log.info("  fit batch %d completado", i + 1)

    log.info("Pasada 1 completada.")

    # Aplicar las transformaciones y guardar preprocessed.csv
    log.info("Transformando datos y guardando preprocessed.csv...")

    available_cols = list(pd.read_csv(INPUT_CSV, nrows=0).columns)
    cols_to_read   = [c for c in (numeric_cols + ohe_cols + label_cols + passthrough) if c in available_cols]

    output_path = OUTPUT_DIR / "preprocessed.csv"
    total_rows  = 0
    first_chunk = True

    for i, chunk in enumerate(pd.read_csv(INPUT_CSV, usecols=cols_to_read, chunksize=CHUNK_SIZE, low_memory=False)):
        parts = []

        # Escalar columnas numéricas con el scaler ya ajustado
        if numeric_cols:
            X_scaled = scaler.transform(chunk[numeric_cols].to_numpy(dtype=np.float64))
            parts.append(pd.DataFrame(X_scaled, columns=numeric_cols, index=chunk.index))

        # OHE: crear una columna binaria por cada valor posible
        for col in ohe_cols:
            mapping = encodings.get(col, {})
            inv_map = {v: k for k, v in mapping.items()}                    # int → string original
            labels  = chunk[col].map(inv_map).fillna("unknown").astype(str) # recuperar el string
            for category in sorted(mapping.keys()):
                parts.append((labels == category).astype(int).rename(f"{col}__{category}"))

        # Label encoding para columnas con muchos valores únicos (se dejan como enteros)
        if label_cols:
            parts.append(chunk[label_cols].reset_index(drop=True))

        # Columnas de listas se pasan sin cambios
        available_passthrough = [c for c in passthrough if c in chunk.columns]
        if available_passthrough:
            parts.append(chunk[available_passthrough].reset_index(drop=True))

        # Unir todas las partes en un solo DataFrame
        result = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
        result.to_csv(output_path, mode="w" if first_chunk else "a", header=first_chunk, index=False)
        first_chunk = False
        total_rows += len(result)
        log.info("  transform batch %d — %d filas", i + 1, total_rows)

    log.info("preprocessed.csv guardado: %d filas", total_rows)

    # Aplicar PCA a las columnas numéricas y guardar aparte
    log.info("Aplicando PCA y guardando pca.csv...")

    pca_col_names = [f"PC{i+1}" for i in range(ipca.n_components_)]
    pca_path = OUTPUT_DIR / "pca.csv"
    total_pca = 0
    first_chunk = True

    for i, chunk in enumerate(pd.read_csv(INPUT_CSV, usecols=numeric_cols, chunksize=CHUNK_SIZE, low_memory=False)):
        X_scaled = scaler.transform(chunk[numeric_cols].to_numpy(dtype=np.float64))
        X_pca    = ipca.transform(X_scaled)
        pca_chunk = pd.DataFrame(X_pca, columns=pca_col_names)
        pca_chunk.to_csv(pca_path, mode="w" if first_chunk else "a", header=first_chunk, index=False)
        first_chunk = False
        total_pca += len(pca_chunk)
        log.info("  pca batch %d — %d filas", i + 1, total_pca)

    log.info("pca.csv guardado: %d filas", total_pca)

    # Guardar varianza explicada por cada componente
    explained = ipca.explained_variance_ratio_
    pca_info  = {
        "components": pca_col_names,
        "explained_variance_ratio": explained.tolist(),
        "cumulative_variance_ratio": np.cumsum(explained).tolist(),
    }
    with open(OUTPUT_DIR / "pca_explained_variance.json", "w") as f:
        json.dump(pca_info, f, indent=2)
    log.info("Varianza explicada acumulada con %d componentes: %.3f",
             ipca.n_components_, pca_info["cumulative_variance_ratio"][-1])

    # Guardar el scaler y el pca para poder usarlos en el futuro
    with open(OUTPUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(OUTPUT_DIR / "pca.pkl", "wb") as f:
        pickle.dump(ipca, f)

    # Guardar las categorías usadas en OHE (para reproducir el encoding)
    ohe_mappings = {col: sorted(encodings[col].keys()) for col in ohe_cols if col in encodings}
    with open(OUTPUT_DIR / "ohe_mappings.json", "w", encoding="utf-8") as f:
        json.dump(ohe_mappings, f, indent=2, ensure_ascii=False)

    log.info("Transformadores guardados: scaler.pkl, pca.pkl, ohe_mappings.json")


if __name__ == "__main__":
    preprocessing()
