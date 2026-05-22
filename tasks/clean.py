import orjson
import logging
import polars as pl
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


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


_cfg = load_config()
CHUNK_SIZE = _cfg.get("general", {}).get("chunk_size", 50_000)
NULL_THRESHOLD = _cfg.get("clean", {}).get("null_threshold", 0.70)
NUMERIC_CATEGORICAL_THRESHOLD = _cfg.get("clean", {}).get("numeric_categorical_threshold", 20)


def classify_numeric(series: pl.Series) -> str:
    """
    Subclasifica una serie numérica en:
      - numeric_categorical -> pocos valores únicos, probablemente una categoría codificada
      - numeric -> cualquier otro valor numérico (discreto o continuo)
    """
    n_unique = series.n_unique()

    if n_unique <= NUMERIC_CATEGORICAL_THRESHOLD:
        return "numeric_categorical"

    return "numeric"


def detect_column_type(series: pl.Series, col_name: str = "", no_list_cols=None):
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

    series = series.drop_nulls()

    # ¿Es numérica?
    if series.dtype.is_numeric():
        return classify_numeric(series), None

    # ¿Se puede convertir a número?
    try:
        numeric_series = series.cast(pl.Float64, strict=False)
        if numeric_series.null_count() == 0:
            return classify_numeric(numeric_series), None
    except Exception:
        pass

    sample = series.cast(pl.String)

    # ¿Solo Y/N?
    if set(sample.str.strip_chars().unique().to_list()) <= {"Y", "N"}:
        return "boolean", None

    # Verificamos que la columna no esté excluida explícitamente de ser lista
    if col_name not in no_list_cols:
        # ¿La mayoría empieza con {? → dict JSON
        starts_with_brace = sample.str.strip_chars().str.starts_with("{").mean()
        if starts_with_brace > 0.5:
            return "dict_json", "dict"

        # ¿La mayoría empieza con [? → lista JSON
        starts_with_bracket = sample.str.strip_chars().str.starts_with("[").mean()
        if starts_with_bracket > 0.5:
            return "list_json", "json"

        # ¿La mayoría tiene comas? → lista CSV
        has_comma = sample.str.contains(",", literal=True).mean()
        if has_comma > 0.3:
            return "list_json", "csv"

    return "categorical", None


def parse_and_explode_chunk(series: pl.Series, hint: str = None) -> pl.DataFrame:
    """
    Analiza las listas dependiendo de la pista (hint), las expande y mantiene el índice de la fila.
    Devuelve un DataFrame: [{"row_id": 0, "value": "A"}, ...]
    """
    series = series.drop_nulls()
    if series.is_empty():
        return pl.DataFrame({"row_id": pl.Series([], dtype=pl.Int64),
                             "value": pl.Series([], dtype=pl.String)})

    def parse_val(val):
        if val is None:
            return None
        if isinstance(val, list):
            return val

        if isinstance(val, str):
            val = val.strip()
            if not val:
                return None

            # Intentar parsear como Python/JSON si es 'dict' o 'json'
            if hint in ("dict", "json"):
                if (val.startswith('[') and val.endswith(']')) or (val.startswith('{') and val.endswith('}')):
                    try:
                        result = ast.literal_eval(val)
                        if isinstance(result, list):
                            return result
                        return [result]  # Si es dict, se envuelve en lista para el explode
                    except (ValueError, SyntaxError):
                        if val.startswith('[') and val.endswith(']'):
                            val = val[1:-1].strip()

            # Parsear como valores separados por comas si es 'csv' (o como fallback)
            if hint == "csv" or ',' in val:
                return [item.strip() for item in val.split(',') if item.strip()]

            # Tratar como un string único
            return [val]

        return [val]

    # Construir DataFrame con row_id antes de explotar para conservar el índice original
    df = pl.DataFrame({
        "row_id": series.to_frame().with_row_index("row_id")["row_id"].cast(pl.Int64),
        "value": series,
    })
    parsed = df.with_columns(
        pl.col("value").map_elements(parse_val, return_dtype=pl.List(pl.String))
    ).drop_nulls("value").explode("value").drop_nulls("value")

    return parsed


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

    for chunk in pl.scan_csv(RAW_CSV, infer_schema_length=0).collect_batches(chunk_size=CHUNK_SIZE):
        nc = chunk.null_count().row(0, named=True)
        if null_counts is None:
            null_counts = nc
            all_columns = chunk.columns
        else:
            null_counts = {col: null_counts[col] + nc[col] for col in all_columns}
        total_rows += len(chunk)

    null_ratio = {col: null_counts[col] / total_rows for col in all_columns}
    # quitamos columnas con mas de 70% de nulos
    drop_columns = [c for c in all_columns if null_ratio[c] > NULL_THRESHOLD]
    keep_columns = [c for c in all_columns if c not in drop_columns]

    log.info("Columnas eliminadas (>%.0f%% nulos): %s", NULL_THRESHOLD * 100, drop_columns)
    log.info("Columnas que se mantienen: %d / %d", len(keep_columns), len(all_columns))

    # detectamos el tipo de cada columna para tratarla adecuadamente
    log.info("Detectando tipos de columnas...")
    sample = pl.read_csv(RAW_CSV, columns=keep_columns, n_rows=10_000, infer_schema_length=0)

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
    # calculamos las estadisticas necesarias para despues hacer imputacion y encoding
    NUMERIC_TYPES = {"numeric"}
    CATEGORICAL_LIKE = {"categorical", "boolean", "numeric_categorical"}

    numeric_values = {col: [] for col, t in type_dict.items() if t in NUMERIC_TYPES}
    category_counts = {col: {} for col, t in type_dict.items() if t in CATEGORICAL_LIKE}

    for i, chunk in enumerate(pl.scan_csv(RAW_CSV, infer_schema_length=0)
                               .select(keep_columns).collect_batches(chunk_size=CHUNK_SIZE)):
        for col in numeric_values:
            vals = chunk[col].drop_nulls().cast(pl.Float64, strict=False).drop_nulls()
            numeric_values[col].extend(vals.to_list())

        for col in category_counts:
            for val in chunk[col].drop_nulls().cast(pl.String).to_list():
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

    for i, chunk in enumerate(pl.scan_csv(RAW_CSV, infer_schema_length=0)
                               .select(keep_columns).collect_batches(chunk_size=CHUNK_SIZE)):

        for col in json_extract_cols:
            if col in chunk.columns:
                # Usar la pista de procesamiento específica para cada columna
                exploded_df = parse_and_explode_chunk(chunk[col], hint=processing_hints.get(col))

                if not exploded_df.is_empty():
                    records = exploded_df.to_dicts()
                    json_str = orjson.dumps(records).decode("utf-8")[1:-1]  # quitar [ y ]
                    if json_str.strip():
                        f = json_file_handles[col]
                        if not is_first_json_chunk[col]:
                            f.write(",\n")
                        f.write(json_str)
                        is_first_json_chunk[col] = False

                chunk = chunk.drop(col)

        # Imputar nulos y aplicar encodings por columna
        fill_exprs = []
        cast_exprs = []
        enc_exprs = []

        for col, dtype in type_dict.items():
            if col not in chunk.columns:
                continue

            if chunk[col].null_count() > 0:
                fv = fill_values.get(col, "")
                fill_exprs.append(pl.col(col).fill_null(pl.lit(fv).cast(pl.String)))

            if dtype in NUMERIC_TYPES:
                cast_exprs.append(
                    pl.col(col).cast(pl.Float64, strict=False).fill_null(fill_values.get(col, 0.0))
                )

        if fill_exprs:
            chunk = chunk.with_columns(fill_exprs)
        if cast_exprs:
            chunk = chunk.with_columns(cast_exprs)

        for col, mapping in encodings.items():
            if col in chunk.columns:
                # Mapear valor de cadena → índice entero
                enc_exprs.append(
                    pl.col(col).cast(pl.String).replace(mapping).alias(col)
                )
        if enc_exprs:
            chunk = chunk.with_columns(enc_exprs)

        if first_chunk:
            chunk.write_csv(output_csv)
            first_chunk = False
        else:
            with open(output_csv, "a") as f:
                f.write(chunk.write_csv(include_header=False))

        total_rows_written += len(chunk)
        log.info("  transform batch %d — %d filas escritas", i + 1, total_rows_written)

    for col, f in json_file_handles.items():
        f.write("\n]\n")
        f.close()
        log.info("Archivo JSON guardado: %s.json", col)

    log.info("clean.csv guardado: %d filas", total_rows_written)

    with open(OUTPUT_DIR / "type_dict.json", "wb") as f:
        f.write(orjson.dumps(type_dict, option=orjson.OPT_INDENT_2))

    with open(OUTPUT_DIR / "encodings.json", "wb") as f:
        f.write(orjson.dumps(encodings, option=orjson.OPT_INDENT_2))

    with open(OUTPUT_DIR / "processing_hints.json", "wb") as f:
        f.write(orjson.dumps(processing_hints, option=orjson.OPT_INDENT_2))

    log.info("Artefactos guardados: type_dict.json, encodings.json, processing_hints.json")


if __name__ == "__main__":
    clean()
