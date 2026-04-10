# TripAdvisor Restaurants Pipeline

Pipeline de datos con Apache Airflow para procesar el dataset de restaurantes europeos de TripAdvisor (~1M filas, 42 columnas).

---

## Estructura

```
proyecto/
├── dags/
│   └── pipeline.py                  # Definición del DAG de Airflow
├── tasks/
│   ├── extract.py
│   ├── clean.py
│   ├── eda.py
│   ├── preprocessing.py
│   └── load.py
├── data/
│   ├── raw/
│   │   └── raw.csv                  # Salida de extract
│   └── processed/
│       ├── clean.csv                # Salida de clean
│       ├── type_dict.json
│       ├── encodings.json
│       ├── processing_hints.json
│       ├── cuisines.json            # Columnas de listas extraídas
│       ├── meals.json
│       ├── top_tags.json
│       ├── original_location.json
│       ├── original_open_hours.json
│       ├── summary_stats.json       # Salida de eda
│       ├── numeric_stats.json
│       ├── preprocessed.csv         # Salida de preprocessing
│       ├── pca.csv
│       ├── scaler.pkl
│       ├── pca.pkl
│       ├── ohe_mappings.json
│       └── pca_explained_variance.json
├── eda/
│   ├── numeric/                     # Histogramas + boxplots
│   ├── categorical/                 # Gráficos de barras
│   ├── boolean/
│   ├── list_json/                   # Heatmaps de co-ocurrencia
│   └── scatters/                    # Matriz de dispersión
├── config.toml                      # Parámetros configurables del pipeline
├── informe.md
├── grafo_pipeline.png
├── pyproject.toml
└── tripadvisor_european_restaurants.csv
```

---

## Requisitos

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) — gestiona las dependencias (incluye Airflow)
- Docker (para Kafka)

## Cómo ejecutar

### 1. Descargar el dataset

Descarga el dataset desde [Kaggle](https://www.kaggle.com/datasets/stefanoleone992/tripadvisor-european-restaurants/data) y coloca el fichero `tripadvisor_european_restaurants.csv` en la raíz del proyecto. La task `extract` lo tomará de ahí y generará `data/raw/raw.csv` automáticamente.

### 2. Instalar dependencias

```bash
cd sd2/proyecto
uv sync
```

### 3. Levantar Airflow

```bash
AIRFLOW_HOME=$(pwd) uv run airflow standalone
```

La UI queda en **http://localhost:8080**. Las credenciales se generan en `simple_auth_manager_passwords.json.generated`.

### 4. Lanzar el pipeline

Desde la UI: activar el toggle del DAG `tripadvisor_pipeline` → **Trigger DAG**.

O desde terminal:

```bash
AIRFLOW_HOME=$(pwd) uv run airflow dags trigger tripadvisor_pipeline
```

### 5. Ejecutar una task individualmente (si se quiere probar)

```bash
uv run python -m tasks.extract
uv run python -m tasks.clean
uv run python -m tasks.eda
uv run python -m tasks.preprocessing
```

> **Nota**: la task `load` requiere un broker de Kafka corriendo

### 6. Levantar Kafka (necesario para la task `load`)

```bash
docker run -d --name kafka -p 9092:9092 apache/kafka:latest
```

Luego crea el topic necesario:

```bash
docker exec kafka /opt/kafka/bin/kafka-topics.sh \
  --create \
  --topic restaurants \
  --bootstrap-server localhost:9092 \
  --partitions 1 \
  --replication-factor 1
```

> Si el contenedor ya existe de una sesión anterior, usa `docker start kafka` en lugar de `docker run`.

Para comprobar que está corriendo:

```bash
docker ps
```

Para pararlo:

```bash
docker stop kafka
```

---

## Pipeline

```
extract → clean → eda → preprocessing → load
```

### extract
Lee el CSV fuente en chunks de 50.000 filas y lo guarda en `data/raw/raw.csv`.

### clean
- Elimina columnas con más del 70% de valores nulos
- Detecta el tipo de cada columna (numérica, booleana, lista, categórica)
- Imputa valores faltantes (mediana para numéricas, moda para el resto)
- Label encoding para columnas categóricas y booleanas
- Guarda `clean.csv`, `type_dict.json`, `encodings.json` en `data/processed/`

### eda
- Genera gráficos de distribución (histograma con curva KDE y boxplot) para cada variable numérica.
- Calcula valores atípicos mediante rango intercuartílico (IQR) y la matriz de correlación de Pearson.
- Crea gráficos de barras legibles para frecuencias de variables categóricas y booleanas, revirtiendo temporalmente el encoding.
- Construye mapas de calor de co-ocurrencia (heatmaps) para los elementos más comunes dentro de las listas JSON.
- Dibuja una matriz de dispersión multivariable utilizando una muestra representativa para evitar el agotamiento de memoria.
- Guarda las imágenes organizadas por tipo de dato en la carpeta `eda/` y exporta la metadata `numeric_stats.json` en `data/processed/`.

### preprocessing
- Normalización con `StandardScaler`
- One-Hot Encoding para columnas categóricas de baja cardinalidad (≤ 15 valores únicos)
- PCA incremental sobre columnas numéricas
- Todo se procesa en batches para no cargar el dataset entero en memoria
- Guarda `preprocessed.csv`, `pca.csv`, `scaler.pkl`, `pca.pkl` en `data/processed/`

### load
Envía cada fila de `preprocessed.csv` a un topic de Kafka como mensaje JSON. Usa un productor con confirmación (`acks=all`) y procesa en sub-batches para controlar el uso de memoria.

---

## Configuración

El archivo `config.toml` centraliza todos los parámetros del pipeline:

| Sección | Clave | Descripción |
|---|---|---|
| `[general]` | `chunk_size` | Filas leídas por batch en todos los tasks |
| `[clean]` | `null_threshold` | Fracción de nulos para eliminar una columna |
| `[clean]` | `numeric_categorical_threshold` | Valores únicos máximos para clasificar como `numeric_categorical` |
| `[eda]` | `top_n_categories` | Categorías mostradas en gráficos de barras |
| `[eda]` | `top_n_cooccurrence` | Elementos en heatmaps de co-ocurrencia |
| `[eda]` | `scatter_sample_rows` | Filas muestreadas para la matriz de dispersión |
| `[eda]` | `plot_dpi` | Resolución de las imágenes generadas |
| `[preprocessing]` | `ohe_max_cardinality` | Cardinalidad máxima para OHE (por encima → label encoding) |
| `[kafka]` | `bootstrap_servers`, `topic`, `batch_size` | Conexión y configuración del productor Kafka |

---

## Dataset

`tripadvisor_european_restaurants.csv` — ~1.083.000 filas, 42 columnas.

Columnas eliminadas en la limpieza por tener demasiados nulos:

| Columna | % nulos |
|---------|---------|
| keywords | 90.8% |
| atmosphere | 75.8% |
| awards | 75.7% |
| price_range | 71.9% |
| features | 70.7% |
