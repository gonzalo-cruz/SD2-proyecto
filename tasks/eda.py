import json
import logging
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from collections import Counter
from scipy.stats import gaussian_kde

PROCESSED_CSV = Path(__file__).parent.parent / "data" / "processed" / "clean.csv"
ARTIFACTS_DIR = Path(__file__).parent.parent / "data" / "processed"
EDA_DIR = Path(__file__).parent.parent / "eda"

CHUNK_SIZE = 50_000

# Suppress harmless matplotlib warning about plotting string-numbers
warnings.filterwarnings("ignore", message=".*categorical units to plot a list of strings.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# Crea carpetas para almacenar los gráficos resultantes
def create_directories():
    for sub_dir in ["numeric", "categorical", "boolean", "list_json", "scatters"]:
        (EDA_DIR / sub_dir).mkdir(parents=True, exist_ok=True)


# Grafica variables numéricas. Detecta atípicos y aplica logaritmo si hay mucha asimetría.
def plot_numeric(data, column, stats_dict):
    q1, q3 = np.percentile(data, 25), np.percentile(data, 75)
    iqr = q3 - q1
    outliers = data[(data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)]
    stats_dict["outliers"][column] = len(outliers)

    # Transformación logarítmica solo para visualización si es muy asimétrica
    data_plot = data
    if abs(pd.Series(data).skew()) > 2.0:
        log.info("  Aplicando logaritmo a '%s' por alta asimetría visual.", column)
        data_plot = np.log1p(data - np.min(data))
        title_suffix = " (Log Transform)"
    else:
        title_suffix = ""

    fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [0.2, 0.8]})
    
    ax_box.boxplot(data_plot, vert=False)
    ax_box.set_title(f"Distribución de {column}{title_suffix}")
    ax_box.set_yticks([])
    
    ax_hist.hist(data_plot, bins=50, density=True, alpha=0.6, edgecolor="black", linewidth=0.5)
    
    # KDE suavizado (bw_method controla la suavidad)
    try:
        x_vals = np.linspace(data_plot.min(), data_plot.max(), 200)
        kde = gaussian_kde(data_plot, bw_method=0.6)  # Curva más suave
        ax_hist.plot(x_vals, kde(x_vals), color="red", linewidth=2, label="Densidad")
    except np.linalg.LinAlgError:
        pass # Ignorar si la varianza es muy cercana a 0
    
    ax_hist.set_xlabel(column)
    ax_hist.set_ylabel("Densidad")
    ax_hist.legend()
    
    plt.tight_layout()
    fig.savefig(EDA_DIR / "numeric" / f"{column}_hist_box.png", dpi=150)
    plt.close(fig)


# Gráfico de barras para variables categóricas o numéricas discretas (numeric_categorical).
def plot_categorical(counts, column, reverse_mapping, col_type, top_n=20):
    # Limita a top_n categorías para evitar colapsar matplotlib en alta cardinalidad
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:top_n]

    # Usar el mapeo de nombres si existe; si no, usar el valor numérico/original
    if reverse_mapping:
        labels = [reverse_mapping.get(str(k), str(k)) for k, v in sorted_counts]
    else:
        labels = [str(k) for k, v in sorted_counts]
        
    values = [v for k, v in sorted_counts]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values, color="skyblue", edgecolor="black")
    
    title_suffix = f" (Top {len(labels)})" if len(counts) > top_n else ""
    ax.set_title(f"Frecuencia de {column}{title_suffix}")
    ax.set_xlabel(column)
    ax.set_ylabel("Conteo")
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    # Guardamos numeric_categorical en la carpeta categórica
    folder = "categorical" if col_type == "numeric_categorical" else col_type
    fig.savefig(EDA_DIR / folder / f"{column}_bar.png", dpi=150)
    plt.close(fig)


# Matriz de co-ocurrencia. La diagonal tiene las frecuencias individuales de cada palabra.
def plot_cooccurrence_heatmap(single_counts, pair_counts, column, top_n=15):
    top_elements = [item for item, count in single_counts.most_common(top_n)]
    n = len(top_elements)
    if n == 0:
        log.warning("    No se encontraron elementos válidos para '%s', se omite el gráfico.", column)
        return
    elif n == 1:
        log.warning("    Solo hay 1 elemento único en '%s', la matriz será de 1x1.", column)

    matrix = np.zeros((n, n))
    matrix[:] = np.nan

    for i in range(n):
        for j in range(n):
            if i >= j:
                if i == j:
                    matrix[i, j] = single_counts[top_elements[i]]
                else:
                    pair = tuple(sorted([top_elements[i], top_elements[j]]))
                    matrix[i, j] = pair_counts.get(pair, 0)

    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.imshow(matrix, cmap="Blues", aspect="auto")

    max_val = np.nanmax(matrix)
    for i in range(n):
        for j in range(n):
            if i >= j:
                val = matrix[i, j]
                if not np.isnan(val) and val > 0:
                    text_color = "white" if val > max_val / 2 else "black"
                    ax.text(j, i, int(val), ha="center", va="center", color=text_color, fontsize=9)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(top_elements, rotation=45, ha="right")
    ax.set_yticklabels(top_elements)
    ax.set_title(f"Matriz de Co-ocurrencia: {column}")

    plt.colorbar(cax, ax=ax)
    plt.tight_layout()
    fig.savefig(EDA_DIR / "list_json" / f"{column}_cooccurrence.png", dpi=150)
    plt.close(fig)


# Lee archivo JSON iterativamente para no llenar la RAM y cuenta ocurrencias/pares por fila
def process_list_json_chunked(filepath):
    single_counts = Counter()
    pair_counts = Counter()
    
    current_row = None
    current_items = []
    items_processed = 0

    if not filepath.exists():
        log.warning("    [ADVERTENCIA] No se encontró el archivo %s. Se ignorará esta variable.", filepath.name)
        return single_counts, pair_counts

    decoder = json.JSONDecoder()
    buffer = ""

    with open(filepath, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(8192) # Leer en trozos de 8KB
            if not chunk:
                break
            buffer += chunk
            
            while buffer:
                # Limpiar caracteres separadores, espacios o corchetes de inicio/fin de la lista JSON
                buffer = buffer.lstrip(" \n\r\t,[]")
                if not buffer:
                    break
                
                try:
                    # raw_decode lee el primer objeto JSON válido que encuentre en el buffer
                    obj, index = decoder.raw_decode(buffer)
                    r_id, val = obj["row_id"], str(obj["value"])
                    
                    items_processed += 1
                    if items_processed % 100_000 == 0:
                        log.info("    ... %d objetos JSON analizados de %s", items_processed, filepath.name)
                    
                    if r_id != current_row:
                        # Fila anterior completada: registrar combinaciones
                        if current_items:
                            uniq = list(set(current_items))
                            for u in uniq: single_counts[u] += 1
                            for p in itertools.combinations(sorted(uniq), 2): pair_counts[p] += 1
                        current_row = r_id
                        current_items = [val]
                    else:
                        current_items.append(val)
                        
                    # Avanzar el buffer saltando el objeto que acabamos de decodificar
                    buffer = buffer[index:]
                except json.JSONDecodeError:
                    # Si el objeto JSON está incompleto, esperamos al siguiente chunk para completar el buffer
                    # Preventivo: evitar que el buffer crezca sin control si el archivo es inválido
                    if len(buffer) > 10_000_000:
                        log.warning(f"El buffer excede los 10MB analizando {filepath.name}. ¿JSON corrupto?")
                        buffer = ""
                    break

    # Procesar la última fila del archivo
    if current_items:
        uniq = list(set(current_items))
        for u in uniq: single_counts[u] += 1
        for p in itertools.combinations(sorted(uniq), 2): pair_counts[p] += 1
        
    return single_counts, pair_counts


# Matriz de dispersión con muestra de datos.
def plot_scatters(numeric_columns, stats_dict):
    if len(numeric_columns) < 2: return
    log.info("Generando gráficos de dispersión con muestra...")
    sample_df = pd.read_csv(PROCESSED_CSV, usecols=numeric_columns, nrows=10_000)
    
    corr_matrix = sample_df.corr(method="pearson")
    stats_dict["pearson_correlation"] = corr_matrix.to_dict()

    n = len(numeric_columns)
    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    if n == 2: axes = np.array(axes).reshape(2, 2)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.text(0.5, 0.5, numeric_columns[i], ha="center", va="center", fontsize=12, fontweight="bold")
                ax.set_axis_off()
            else:
                ax.scatter(sample_df[numeric_columns[j]], sample_df[numeric_columns[i]], alpha=0.3, s=10)
                pearson_val = corr_matrix.loc[numeric_columns[i], numeric_columns[j]]
                ax.set_title(f"Pearson: {pearson_val:.2f}", fontsize=10)
            
            if i == n - 1: ax.set_xlabel(numeric_columns[j])
            if j == 0: ax.set_ylabel(numeric_columns[i])
                
    plt.tight_layout()
    fig.savefig(EDA_DIR / "scatters" / "matriz_dispersion.png", dpi=150)
    plt.close(fig)


# Orquesta el análisis. Construye el resumen iterativamente por chunks y llama a graficadores.
def eda():
    create_directories()

    with open(ARTIFACTS_DIR / "type_dict.json", "r", encoding="utf-8") as f:
        type_dict = json.load(f)
    with open(ARTIFACTS_DIR / "encodings.json", "r", encoding="utf-8") as f:
        encodings = json.load(f)

    reverse_encodings = {col: {str(v): k for k, v in mapping.items()} for col, mapping in encodings.items()}

    # Agrupación por tipos
    numeric_cols = [c for c, t in type_dict.items() if t == "numeric"]
    categorical_cols = [c for c, t in type_dict.items() if t in ("categorical", "boolean", "numeric_categorical")]
    list_cols = [c for c, t in type_dict.items() if t == "list_json"]

    numeric_stats = {"outliers": {}, "pearson_correlation": {}}
    
    # Inicialización del resumen (summary)
    summary_stats = {col: {"type": type_dict[col], "count": 0} for col in type_dict.keys()}
    for col in numeric_cols:
        summary_stats[col].update({"min": float('inf'), "max": float('-inf'), "sum": 0.0})
    for col in categorical_cols:
        summary_stats[col]["unique_values_set"] = set()

    cat_counts = {col: Counter() for col in categorical_cols}

    log.info("Procesando CSV en chunks para resumen y categorías...")
    total_rows_processed = 0
    
    for i, chunk in enumerate(pd.read_csv(PROCESSED_CSV, usecols=numeric_cols + categorical_cols, chunksize=CHUNK_SIZE)):
        
        # Resumen Numéricas
        for col in numeric_cols:
            if col in chunk:
                c_data = chunk[col].dropna()
                summary_stats[col]["count"] += len(c_data)
                if not c_data.empty:
                    summary_stats[col]["min"] = float(min(summary_stats[col]["min"], c_data.min()))
                    summary_stats[col]["max"] = float(max(summary_stats[col]["max"], c_data.max()))
                    summary_stats[col]["sum"] += float(c_data.sum())
        
        # Resumen y Conteo Categóricas
        for col in categorical_cols:
            if col in chunk:
                c_data = chunk[col].dropna().astype(str)
                summary_stats[col]["count"] += len(c_data)
                summary_stats[col]["unique_values_set"].update(c_data.unique())
                cat_counts[col].update(c_data)
                
        total_rows_processed += len(chunk)
        log.info("  batch %d completado — %d filas analizadas", i + 1, total_rows_processed)

    # Calcular medias finales y limpiar sets
    for col in numeric_cols:
        count = summary_stats[col]["count"]
        summary_stats[col]["mean"] = summary_stats[col]["sum"] / count if count > 0 else 0
        del summary_stats[col]["sum"]  # Borrar suma temporal
    
    for col in categorical_cols:
        summary_stats[col]["num_unique_values"] = len(summary_stats[col]["unique_values_set"])
        del summary_stats[col]["unique_values_set"]  # El set no es serializable en JSON

    # Graficar Categóricas
    log.info("Graficando variables categóricas...")
    for col in categorical_cols:
        is_num_cat = type_dict[col] == "numeric_categorical"
        rev_map = None if is_num_cat else reverse_encodings.get(col)
        plot_categorical(dict(cat_counts[col]), col, rev_map, type_dict[col])

    # Graficar Numéricas y detectar atípicos (leemos columna completa ya que cabe en RAM)
    log.info("Graficando variables numéricas...")
    for col in numeric_cols:
        col_data = pd.read_csv(PROCESSED_CSV, usecols=[col])[col].dropna()
        plot_numeric(col_data.values, col, numeric_stats)

    # Procesar listas JSON y añadirlas al Summary
    log.info("Procesando listas JSON iterativamente...")
    for col in list_cols:
        log.info("  analizando archivo: %s.json", col)
        filepath = ARTIFACTS_DIR / f"{col}.json"
        single, pair = process_list_json_chunked(filepath)
        
        # Añadir al Summary
        summary_stats[col]["total_items_parsed"] = sum(single.values())
        summary_stats[col]["num_unique_elements"] = len(single)
        
        plot_cooccurrence_heatmap(single, pair, col, top_n=15)

    plot_scatters(numeric_cols, numeric_stats)
    
    # Guardar ambos artefactos JSON
    with open(ARTIFACTS_DIR / "numeric_stats.json", "w", encoding="utf-8") as f:
        json.dump(numeric_stats, f, indent=2, ensure_ascii=False)
        
    with open(ARTIFACTS_DIR / "summary_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
        
    log.info("Artefactos guardados: numeric_stats.json, summary_stats.json")
    log.info("EDA finalizado con éxito.")


if __name__ == "__main__":
    eda()