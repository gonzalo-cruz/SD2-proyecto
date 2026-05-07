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
│   ├── load.py                      # BashOperator que lanza producer.py
│   └── producer.py                  # Productor Kafka (confluent_kafka + orjson)
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
├── kafka-producer-confluent.py      # Demo productor Kafka (topic purchases)
├── kafka-consumer-confluent.py      # Demo consumidor Kafka (topic purchases)
├── struct_kafka_consumer_local.py   # Consumidor Spark Structured Streaming (demo)
├── queries.py                       # Consultas Spark Structured Streaming sobre restaurants
├── docker-compose.yml               # Kafka (Confluent 7.6, KRaft) vía Docker Compose
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
- Java 21 (para Spark Structured Streaming)

## Cómo ejecutar

### 1. Descargar el dataset

Descarga el dataset desde [Kaggle](https://www.kaggle.com/datasets/stefanoleone992/tripadvisor-european-restaurants/data) y coloca el fichero `tripadvisor_european_restaurants.csv` en la raíz del proyecto. La task `extract` lo cogerá de ahí y generará `data/raw/raw.csv` automáticamente.

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
docker compose up -d
```

> Si el contenedor ya existe de una sesión anterior, usa `docker compose start` en lugar de `docker compose up -d`.

Para comprobar que está corriendo:

```bash
docker compose ps
```

Para pararlo:

```bash
docker compose stop
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
Lanza `tasks/producer.py` como subproceso vía `BashOperator` de Airflow. El productor:
- Envía primero el esquema de columnas al topic `restaurants_schema`
- Serializa cada fila de `preprocessed.csv` con `orjson` (más rápido que `json` estándar)
- Usa `confluent_kafka` con confirmación (`acks=all`)
- Maneja backpressure con reintentos automáticos ante `BufferError`
- Delivery callback asíncrono para contabilizar enviados y errores

---

## Spark Structured Streaming

`struct_kafka_consumer_local.py` demuestra el consumo básico del topic `restaurants` usando un sink en memoria para consultas SQL interactivas.

`queries.py` implementa cuatro consultas analíticas sobre el mismo stream, volcando cada resultado a un fichero de texto. Los valores label-encoded (`restaurant_name`, `country`, `city`) se decodifican en tiempo real mediante un join broadcast con los diccionarios de `encodings.json`.

| Query | Modo | Descripción |
|---|---|---|
| 1 | `append` | Campos clave de cada mensaje: nombre, país, ciudad y cocina |
| 2 | `complete` | Conteo de restaurantes por país |
| 3 | `complete` | Ranking de cocinas más frecuentes |
| 4 | `append` | Restaurantes con valoración 3.5 estrellas |

Los resultados se guardan en `salida1.txt` … `salida4.txt`.

### Requisitos adicionales

Crear un entorno virtual con PySpark 4.1.1 (requiere Python 3.11+):

```bash
uv venv pyspark-411 --python 3.11
source pyspark-411/bin/activate
uv pip install "pyspark[connect]==4.1.1"
```

### Ejecutar

Asegúrate de que Kafka tiene datos cargados (`uv run python tasks/producer.py`) y activa el entorno `pyspark-411`:

```bash
source pyspark-411/bin/activate
python struct_kafka_consumer_local.py   # demo memoria
python queries.py                       # consultas analíticas → salida[1-4].txt
```

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
| `[kafka]` | `bootstrap_servers`, `topic` | Conexión y topic del productor Kafka |

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
