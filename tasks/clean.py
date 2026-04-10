import json
import logging
import pandas as pd
import numpy as np
import ast
from pathlib import Path

# Manejo de TOML para la configuración
try:
    import tomllib  # Built-in en Python 3.11+
    TOML_MODE = "rb"
except ImportError:
    try:
        import tomli as tomllib
        TOML_MODE = "rb"
    except ImportError:
        try:
            import toml as tomllib  # Fallback si está instalado
            TOML_MODE = "r"
        except ImportError:
            tomllib = None

# Rutas
RAW_CSV = Path(__file__).parent.parent / "data" / "raw" / "raw.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"
CONFIG_FILE = Path(__file__).parent.parent / "config.toml"

CHUNK_SIZE = 50_000
NULL_THRESHOLD = 0.70  # columnas con más del 70% de nulos se eliminan

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


NUMERIC_CATEGORICAL_THRESHOLD = 20  # columnas numéricas con <= 20 valores únicos → numeric_categorical


def load_config():
    """Carga el archivo de configuración TOML si existe."""
    if not CONFIG_FILE.exists():
        return {}
    if tomllib is None:
        log.warning("No hay módulo TOML disponible (requiere Python 3.11+ o instalar 'tomli'/'toml'). Ignorando config.toml.")
        return {}
    try:
        with open(CONFIG_FILE, TOML_MODE) as f:
            return tomllib.load(f)
    except Exception as e:
        log.warning("Error al leer config.toml: %s", e)
        return {}


def classify_numeric(series):
    """
    Subclasifica una serie numérica en:
      - numeric_categorical -> pocos valores únicos, probablemente una categoría codificada
      - numeric -> cualquier otro valor numérico (discreto o continuo)
    """
    n_unique = series.nunique()

    if n_unique <= NUMERIC_CATEGORICAL_THRESHOLD:
        return "numeric_categorical"

    return "numeric"


def detect_column_type(series, col_name="", no_list_cols=None):
    """
    Clasifica una columna devolviendo una tupla (tipo_principal, pista_de_procesamiento):
      - numeric -> (numeric, None)
      - numeric_categorical -> (numeric_categorical, None)
      - boolean    -> (boolean, None)
      - dict_json  -> (dict_json, "dict")
      - list_json  -> (list_json, "json" / "csv")
      - categorical -> (categorical, None)
    """
    if no_list_cols is None:
        no_list_cols = []

    series = series.dropna()

    # ¿Es numérica?
    if pd.api.types.is_numeric_dtype(series):
        return classify_numeric(series), None

    # ¿Se puede convertir a número?
    try:
        numeric_series = pd.to_numeric(series)
        return classify_numeric(numeric_series), None
    except (ValueError, TypeError):
        pass

    sample = series.astype(str)

    # ¿Solo Y/N?
    if set(sample.str.strip().unique()) <= {"Y", "N"}:
        return "boolean", None

    # Verificamos que la columna no esté excluida explícitamente de ser lista
    if col_name not in no_list_cols:
        # ¿La mayoría empieza con {? → dict JSON
        starts_with_brace = sample.str.strip().str[:1].isin(["{"]).mean()
        if starts_with_brace > 0.5:
            return "dict_json", "dict"

        # ¿La mayoría empieza con [? → lista JSON
        starts_with_bracket = sample.str.strip().str[:1].isin(["["]).mean()
        if starts_with_bracket > 0.5:
            return "list_json", "json"

        # ¿La mayoría tiene comas? → lista CSV
        has_comma = sample.str.contains(",", regex=False).mean()
        if has_comma > 0.3:
            return "list_json", "csv"

    return "categorical", None


def parse_and_explode_chunk(series: pd.Series, hint: str = None) -> pd.DataFrame:
    """
    Analiza las listas dependiendo de la pista (hint), las expande y mantiene el índice de la fila.
    Devuelve un DataFrame: [{"row_id": 0, "value": "A"}, ...]
    """
    series = series.dropna()
    if series.empty:
        return pd.DataFrame(columns=["row_id", "value"])

    def parse_val(val):
        if pd.isna(val):
            return None
        if isinstance(val, list):
            return val
            
        if isinstance(val, str):
            val = val.strip()
            if not val:
                return None
            
            # 1. Intentar parsear como Python/JSON si es 'dict' o 'json'
            if hint in ("dict", "json"):
                if (val.startswith('[') and val.endswith(']')) or (val.startswith('{') and val.endswith('}')):
                    try:
                        result = ast.literal_eval(val)
                        if isinstance(result, list):
                            return result
                        return [result] # Si es dict, se envuelve en lista para el explode
                    except (ValueError, SyntaxError):
                        if val.startswith('[') and val.endswith(']'):
                            val = val[1:-1].strip()
            
            # 2. Parsear como valores separados por comas si es 'csv' (o como fallback)
            if hint == "csv" or ',' in val:
                return [item.strip() for item in val.split(',') if item.strip()]
            
            # 3. Tratar como un string único
            return [val]
            
        return [val]

    parsed = series.apply(parse_val).dropna()
    exploded = parsed.explode().dropna()
    return exploded.reset_index(name="value").rename(columns={"index": "row_id"})


def clean():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()
    no_list_cols = config.get("no_list", [])
    if no_list_cols:
        log.info("Columnas excluidas de detección de listas: %s", no_list_cols)

    log.info("Calculando porcentaje de nulos por columna...")

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

    drop_columns = list(null_ratio[null_ratio > NULL_THRESHOLD].index)
    keep_columns = [c for c in all_columns if c not in drop_columns]

    log.info("Columnas eliminadas (>%.0f%% nulos): %s", NULL_THRESHOLD * 100, drop_columns)
    log.info("Columnas que se mantienen: %d / %d", len(keep_columns), len(all_columns))

    log.info("Detectando tipos de columnas...")
    sample = pd.read_csv(RAW_CSV, usecols=keep_columns, nrows=10_000, low_memory=False)
    
    type_dict = {}
    processing_hints = {}
    
    for col in keep_columns:
        col_type, hint = detect_column_type(sample[col], col_name=col, no_list_cols=no_list_cols)
        type_dict[col] = col_type
        if hint:
            processing_hints[col] = hint

    from collections import Counter
    log.info("Tipos detectados: %s", dict(Counter(type_dict.values())))

    log.info("Paso 3: calculando estadísticas para imputación y encoding...")

    NUMERIC_TYPES = {"numeric"}
    CATEGORICAL_LIKE = {"categorical", "boolean", "numeric_categorical"}

    numeric_values = {col: [] for col, t in type_dict.items() if t in NUMERIC_TYPES}
    category_counts = {col: {} for col, t in type_dict.items() if t in CATEGORICAL_LIKE}

    for i, chunk in enumerate(pd.read_csv(RAW_CSV, usecols=keep_columns, chunksize=CHUNK_SIZE, low_memory=False)):
        for col in numeric_values:
            numeric_values[col].extend(chunk[col].dropna().tolist())

        for col in category_counts:
            for val in chunk[col].dropna().astype(str):
                category_counts[col][val] = category_counts[col].get(val, 0) + 1

        log.info("  stats batch %d completado", i + 1)

    fill_values = {}
    for col, dtype in type_dict.items():
        if dtype in NUMERIC_TYPES:
            vals = numeric_values.get(col, [])
            fill_values[col] = float(np.median(vals)) if vals else 0.0
        elif dtype in CATEGORICAL_LIKE:
            counts = category_counts.get(col, {})
            fill_values[col] = max(counts, key=counts.get) if counts else ""
        else:
            fill_values[col] = "" 

    encodings = {}
    for col, counts in category_counts.items():
        sorted_values = sorted(counts.keys())
        encodings[col] = {val: idx for idx, val in enumerate(sorted_values)}

    log.info("Transformando datos y guardando clean.csv y archivos JSON...")

    output_csv = OUTPUT_DIR / "clean.csv"
    
    # Extraer columnas que son listas o diccionarios complejos para guardarlas en JSON
    json_extract_cols = [col for col, dtype in type_dict.items() if dtype in ("list_json", "dict_json")]
    
    json_file_handles = {}
    is_first_json_chunk = {col: True for col in json_extract_cols}

    for col in json_extract_cols:
        f = open(OUTPUT_DIR / f"{col}.json", "w", encoding="utf-8")
        f.write("[\n")
        json_file_handles[col] = f

    total_rows_written = 0
    first_chunk = True

    for i, chunk in enumerate(pd.read_csv(RAW_CSV, usecols=keep_columns, chunksize=CHUNK_SIZE, low_memory=False)):

        for col in json_extract_cols:
            if col in chunk.columns:
                # Usar la pista de procesamiento específica para cada columna
                exploded_df = parse_and_explode_chunk(chunk[col], hint=processing_hints.get(col))

                if not exploded_df.empty:
                    json_str = exploded_df.to_json(orient="records", force_ascii=False)[1:-1]
                    if json_str.strip():
                        f = json_file_handles[col]
                        if not is_first_json_chunk[col]:
                            f.write(",\n")
                        f.write(json_str)
                        is_first_json_chunk[col] = False

                chunk = chunk.drop(columns=[col])

        for col, dtype in type_dict.items():
            if col in chunk.columns and chunk[col].isna().any():
                chunk[col] = chunk[col].fillna(fill_values.get(col, ""))

        for col, dtype in type_dict.items():
            if dtype in NUMERIC_TYPES and col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(fill_values.get(col, 0))

        for col, mapping in encodings.items():
            if col in chunk.columns:
                chunk[col] = chunk[col].astype(str).map(mapping)

        chunk.to_csv(output_csv, mode="w" if first_chunk else "a", header=first_chunk, index=False)
        first_chunk = False
        total_rows_written += len(chunk)
        log.info("  transform batch %d — %d filas escritas", i + 1, total_rows_written)

    for col, f in json_file_handles.items():
        f.write("\n]\n")
        f.close()
        log.info("Archivo JSON guardado: %s.json", col)

    log.info("clean.csv guardado: %d filas", total_rows_written)

    with open(OUTPUT_DIR / "type_dict.json", "w", encoding="utf-8") as f:
        json.dump(type_dict, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_DIR / "encodings.json", "w", encoding="utf-8") as f:
        json.dump(encodings, f, indent=2, ensure_ascii=False)
        
    with open(OUTPUT_DIR / "processing_hints.json", "w", encoding="utf-8") as f:
        json.dump(processing_hints, f, indent=2, ensure_ascii=False)

    log.info("Artefactos guardados: type_dict.json, encodings.json, processing_hints.json")


if __name__ == "__main__":
    clean()