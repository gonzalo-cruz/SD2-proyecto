import json
import logging
import ast
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from scipy.stats import gaussian_kde

PROCESSED_CSV = Path(__file__).parent.parent / "data" / "processed" / "clean.csv"
ARTIFACTS_DIR = Path(__file__).parent.parent / "data" / "processed"
EDA_DIR = Path(__file__).parent.parent / "eda"

CHUNK_SIZE = 50_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# Creación de directorios para guardar los gráficos.
# 1. Itera sobre una lista de nombres de subcarpetas necesarias.
# 2. Crea las carpetas dentro del directorio principal de EDA si no existen.
def create_directories():
    for sub_dir in ["numeric", "categorical", "boolean", "list_json", "scatters"]:
        (EDA_DIR / sub_dir).mkdir(parents=True, exist_ok=True)


# Graficación de variables numéricas, detección de valores atípicos y registro de metadatos.
# 1. Calcula los cuartiles (Q1, Q3) y el rango intercuartílico (IQR).
# 2. Establece los límites inferior y superior para aislar, contar y registrar los valores atípicos en el diccionario.
# 3. Crea una figura con dos subgráficas: un boxplot horizontal y un histograma de densidad.
# 4. Superpone la curva de densidad estimada (KDE) sobre el histograma.
# 5. Guarda la figura generada en el directorio de variables numéricas.
def plot_numeric(data, column, stats_dict):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    log.info("Variable '%s' - Cantidad de valores atípicos: %d", column, len(outliers))
    
    # Guardar el conteo de atípicos en el diccionario de estadísticas
    stats_dict["outliers"][column] = len(outliers)

    fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [0.2, 0.8]})
    
    ax_box.boxplot(data, vert=False)
    ax_box.set_title(f"Distribución de {column}")
    ax_box.set_yticks([])
    
    ax_hist.hist(data, bins=50, density=True, alpha=0.6, edgecolor="black", linewidth=0.5)
    
    x_vals = np.linspace(data.min(), data.max(), 200)
    kde = gaussian_kde(data)
    ax_hist.plot(x_vals, kde(x_vals), color="red", linewidth=2, label="Curva de Densidad")
    
    ax_hist.set_xlabel(column)
    ax_hist.set_ylabel("Densidad")
    ax_hist.legend()
    
    plt.tight_layout()
    fig.savefig(EDA_DIR / "numeric" / f"{column}_hist_box.png", dpi=150)
    plt.close(fig)


# Graficación de las variables categóricas o booleanas.
# 1. Recupera las etiquetas originales usando el mapeo inverso (si está disponible).
# 2. Extrae las frecuencias y etiquetas correspondientes.
# 3. Construye un gráfico de barras coloreado y ajusta la rotación del eje X para mayor legibilidad.
# 4. Guarda la imagen generada en la carpeta correspondiente a su tipo.
def plot_categorical(counts, column, reverse_mapping, col_type):
    if reverse_mapping:
        labels = [reverse_mapping.get(str(k), str(k)) for k in counts.keys()]
    else:
        labels = list(counts.keys())
        
    values = list(counts.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values, color="skyblue", edgecolor="black")
    
    ax.set_title(f"Frecuencia de {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Conteo")
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    fig.savefig(EDA_DIR / col_type / f"{column}_bar.png", dpi=150)
    plt.close(fig)


# Generación de matriz de co-ocurrencia triangular para listas JSON.
# 1. Selecciona los elementos más frecuentes basándose en los conteos individuales.
# 2. Inicializa una matriz vacía rellenada con NaNs para ocultar el triángulo superior.
# 3. Rellena la diagonal con la frecuencia total del elemento y el triángulo inferior con la frecuencia de los pares.
# 4. Dibuja el mapa de calor usando imshow y superpone los valores numéricos sobre los colores.
# 5. Guarda la gráfica final.
def plot_cooccurrence_heatmap(single_counts, pair_counts, column, top_n=15):
    top_elements = [item for item, count in single_counts.most_common(top_n)]
    n = len(top_elements)

    if n == 0:
        return

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


# Generación de matriz de dispersión y guardado de matriz de correlación.
# 1. Carga una muestra de 10,000 filas para evitar el agotamiento de memoria RAM.
# 2. Calcula la matriz de correlación de Pearson y la guarda en el diccionario de metadatos.
# 3. Construye una cuadrícula combinando todas las variables numéricas e inserta la correlación en el título.
# 4. Genera un gráfico de dispersión en cada intersección, exceptuando la diagonal principal.
# 5. Exporta la matriz resultante al directorio de gráficos de dispersión.
def plot_scatters(numeric_columns, stats_dict):
    log.info("Generando gráficos de dispersión con una muestra de datos...")
    sample_df = pd.read_csv(PROCESSED_CSV, usecols=numeric_columns, nrows=10_000)
    
    n = len(numeric_columns)
    if n < 2:
        return

    corr_matrix = sample_df.corr(method="pearson")
    
    # Guardar la matriz de correlación en el diccionario (convertida a dict nativo de Python)
    stats_dict["pearson_correlation"] = corr_matrix.to_dict()

    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    
    if n == 2:
        axes = np.array(axes).reshape(2, 2)

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
            
            if i == n - 1:
                ax.set_xlabel(numeric_columns[j])
            if j == 0:
                ax.set_ylabel(numeric_columns[i])
                
    plt.tight_layout()
    fig.savefig(EDA_DIR / "scatters" / "matriz_dispersion.png", dpi=150)
    plt.close(fig)


# Función principal que ejecuta el Análisis Exploratorio de Datos (EDA).
# 1. Llama a la creación de los directorios de salida.
# 2. Carga los artefactos (tipos de datos y mapeo de categorías) generados en la fase de limpieza.
# 3. Invierte el diccionario de categorías para que los gráficos muestren texto en lugar de números.
# 4. Inicializa un diccionario general para almacenar las estadísticas de las variables numéricas.
# 5. Procesa y grafica las variables numéricas leyendo el CSV columna por columna y registrando atípicos.
# 6. Procesa las variables categóricas, booleanas y listas en lotes (chunks).
# 7. Cuenta apariciones individuales y pares (co-ocurrencias) para las variables tipo lista.
# 8. Genera gráficos correspondientes y la matriz de dispersión multivariable.
# 9. Guarda las estadísticas numéricas finales en un archivo JSON.
def eda():
    create_directories()

    with open(ARTIFACTS_DIR / "type_dict.json", "r", encoding="utf-8") as f:
        type_dict = json.load(f)
        
    with open(ARTIFACTS_DIR / "encodings.json", "r", encoding="utf-8") as f:
        encodings = json.load(f)

    reverse_encodings = {}
    for col, mapping in encodings.items():
        reverse_encodings[col] = {str(v): k for k, v in mapping.items()}

    numeric_cols = [c for c, t in type_dict.items() if t == "numeric"]
    categorical_cols = [c for c, t in type_dict.items() if t in ["categorical", "boolean"]]
    list_cols = [c for c, t in type_dict.items() if t == "list_json"]

    # Diccionario para almacenar la metadata de variables numéricas
    numeric_stats = {
        "outliers": {},
        "pearson_correlation": {}
    }

    log.info("Procesando variables numéricas...")
    for col in numeric_cols:
        col_data = pd.read_csv(PROCESSED_CSV, usecols=[col])[col].dropna()
        plot_numeric(col_data.values, col, numeric_stats)

    log.info("Procesando variables categóricas y listas JSON...")
    cat_counts = {col: Counter() for col in categorical_cols}
    single_list_counts = {col: Counter() for col in list_cols}
    pair_list_counts = {col: Counter() for col in list_cols}

    for chunk in pd.read_csv(PROCESSED_CSV, usecols=categorical_cols + list_cols, chunksize=CHUNK_SIZE):
        for col in categorical_cols:
            cat_counts[col].update(chunk[col].dropna().astype(str))
            
        for col in list_cols:
            for item in chunk[col].dropna():
                try:
                    elements = ast.literal_eval(item)
                    if isinstance(elements, dict):
                        elements = list(elements.keys())
                    
                    if isinstance(elements, list):
                        unique_elements = list(set([str(e) for e in elements]))
                        
                        for el in unique_elements:
                            single_list_counts[col][el] += 1
                            
                        for pair in itertools.combinations(sorted(unique_elements), 2):
                            pair_list_counts[col][pair] += 1
                except (ValueError, SyntaxError):
                    pass

    for col in categorical_cols:
        col_type = type_dict[col]
        plot_categorical(dict(cat_counts[col]), col, reverse_encodings.get(col, None), col_type)

    for col in list_cols:
        plot_cooccurrence_heatmap(single_list_counts[col], pair_list_counts[col], col, top_n=15)

    # Generar matriz de dispersión y añadir la correlación al diccionario
    plot_scatters(numeric_cols, numeric_stats)
    
    # Guardar el artefacto de estadísticas numéricas
    with open(ARTIFACTS_DIR / "numeric_stats.json", "w", encoding="utf-8") as f:
        json.dump(numeric_stats, f, indent=2, ensure_ascii=False)
        
    log.info("Artefacto guardado: numeric_stats.json")
    log.info("EDA finalizado. Gráficos guardados en %s", EDA_DIR)


if __name__ == "__main__":
    eda()