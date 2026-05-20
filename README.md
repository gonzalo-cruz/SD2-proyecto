# TripAdvisor Restaurants Pipeline

Pipeline de datos con Apache Airflow para procesar el dataset de restaurantes europeos de TripAdvisor (~1M filas, 42 columnas), entrenar un modelo KMeans con Spark ML y exponerlo en una aplicación Streamlit con soporte de streaming en tiempo real via Kafka.

---

## Estructura

```
proyecto/
├── app.py                           # Punto de entrada de la app Streamlit
├── config.py                        # Rutas y constantes globales
├── loader.py                        # Carga de datos para la app (cached)
├── recommendations.py               # Logica de recomendacion KMeans
├── streaming_score.py               # Consumidor Spark Structured Streaming
├── train_model.py                   # Entrenamiento KMeans con Spark ML
│
├── pages/                           # Pestanas de la app Streamlit
│   ├── explore.py                   #   Tab "Explorar": filtros, busqueda y recomendaciones
│   └── live.py                      #   Tab "En vivo": stream en tiempo real con auto-refresh
│
├── ui/                              # Componentes de interfaz reutilizables
│   ├── styles.py                    #   CSS global (fuente, colores, layout)
│   ├── filters.py                   #   Widgets de filtrado y logica de apply_filters()
│   └── tables.py                    #   Construccion de tablas y columnas para Streamlit
│
├── dags/                            # DAGs de Apache Airflow
│   └── pipeline.py                  #   DAG principal: extract→clean→eda→preprocessing→load
│
├── tasks/                           # Tareas del pipeline ETL (una por fichero)
│   ├── extract.py                   #   Lee el CSV fuente en chunks y guarda raw.csv
│   ├── clean.py                     #   Limpieza, imputacion y label encoding
│   ├── eda.py                       #   Visualizaciones automaticas por tipo de columna
│   ├── preprocessing.py             #   Normalizacion, OHE y PCA incremental
│   ├── load.py                      #   BashOperator que lanza producer.py
│   └── producer.py                  #   Productor Kafka: serializa preprocessed.csv con orjson
│
├── scripts/                         # Scripts de exploracion y demos (no son parte del pipeline)
│   ├── queries.py                   #   6 consultas Spark Structured Streaming de exploracion
│   ├── kafka-producer-confluent.py  #   Demo standalone de productor Kafka
│   ├── kafka-consumer-confluent.py  #   Demo standalone de consumidor Kafka
│   └── struct_kafka_consumer_local.py
│
├── docs/                            # Documentacion del proyecto
│   ├── memoria.pdf                  #   PDF memoria
│   ├── informe.md                   #   Descripcion del grafo del pipeline
│   └── grafo_pipeline.png           #   Imagen del grafo de Airflow
│
├── data/                            # Datos generados por el pipeline (no en git)
│   ├── raw/                         #   raw.csv generado por extract
│   ├── processed/                   #   clean.csv, preprocessed.csv y artefactos ETL
│   └── streaming/                   #   results.csv generado por streaming_score.py
│
├── models/                          # Artefactos del modelo (generados por train_model.py)
│   ├── kmeans_spark/                #   Modelo KMeans serializado (formato Spark ML)
│   ├── cluster_assignments.parquet  #   cluster + dist_to_centroid por restaurante
│   ├── feature_cols.json            #   Lista de las 81 features usadas en el entrenamiento
│   ├── best_k.json                  #   K optimo cacheado (eliminar para repetir busqueda)
│   └── k_selection.png              #   Grafica silhouette score vs K y metodo del codo
│
├── docker-compose.yml               # Kafka 
├── config.toml                      # Parametros configurables del pipeline
├── pyproject.toml                   # Dependencias del proyecto (uv)
└── tripadvisor_european_restaurants.csv  # Dataset fuente (descargar de Kaggle)
```

---

## Requisitos

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- Docker
- Java 21 (`sudo apt install openjdk-21-jdk`)

---

## Instalacion

```bash
git clone <repo> && cd proyecto
uv sync
```

---

## Ejecucion completa

### 1. Dataset

Descarga `tripadvisor_european_restaurants.csv` desde [Kaggle](https://www.kaggle.com/datasets/stefanoleone992/tripadvisor-european-restaurants/data) y colócalo en la raiz del proyecto.

### 2. Pipeline ETL (Airflow)

```bash
AIRFLOW_HOME=$(pwd) uv run airflow standalone
```

Abre **http://localhost:8080** — credenciales en `simple_auth_manager_passwords.json.generated`.
Activa y dispara el DAG `tripadvisor_pipeline`.

O ejecutar las tasks manualmente:

```bash
uv run python -m tasks.extract
uv run python -m tasks.clean
uv run python -m tasks.eda
uv run python -m tasks.preprocessing
```

### 3. Entrenar el modelo KMeans

```bash
source pyspark-411/bin/activate   # entorno con PySpark 4.1.1
python train_model.py
```

Genera `models/kmeans_spark/` y `models/cluster_assignments.parquet`.

> Si el entorno pyspark-411 no existe:
> ```bash
> uv venv pyspark-411 --python 3.11
> source pyspark-411/bin/activate
> uv pip install pyspark==4.1.1 orjson confluent-kafka polars streamlit pandas numpy
> ```

### 4. Levantar Kafka

```bash
docker compose up -d
```

### 5. Lanzar la app + streaming (3 terminales)

**Terminal 1 — App:**
```bash
source pyspark-411/bin/activate
streamlit run app.py
```

**Terminal 2 — Productor Kafka:**
```bash
source pyspark-411/bin/activate
python tasks/producer.py
```

**Terminal 3 — Spark Streaming:**
```bash
source pyspark-411/bin/activate
python streaming_score.py
```

La app queda disponible en **http://localhost:8501**.

---

## Pipeline ETL

```
extract → clean → eda → preprocessing → load
```

### extract
Lee el CSV fuente en chunks de 50.000 filas y lo guarda en `data/raw/raw.csv`.

### clean
- Elimina columnas con mas del 70% de valores nulos
- Detecta el tipo de cada columna (numerica, booleana, lista, categorica)
- Imputa valores faltantes (mediana para numericas, moda para el resto)
- Label encoding para columnas categoricas y booleanas
- Guarda `clean.csv`, `type_dict.json`, `encodings.json` en `data/processed/`

### eda
- Histogramas, boxplots y correlacion de Pearson para variables numericas
- Graficos de barras para categoricas y booleanas
- Heatmaps de co-ocurrencia para columnas de listas JSON
- Imagenes guardadas en `eda/` organizadas por tipo de dato

### preprocessing
- Normalizacion con `StandardScaler`
- One-Hot Encoding para categoricas de baja cardinalidad (<= 15 valores unicos)
- PCA incremental sobre columnas numericas
- Procesado en batches para no cargar el dataset entero en memoria
- Guarda `preprocessed.csv`, `type_dict_encoding.json` en `data/processed/`

### load
BashOperator que ejecuta `tasks/producer.py`:
- Envia el esquema de columnas al topic `restaurants_schema`
- Serializa cada fila de `preprocessed.csv` con `orjson`
- Usa `confluent_kafka` con confirmacion (`acks=all`)

---

## Modelo KMeans (`train_model.py`)

1. **Seleccion de K**: evalua K ∈ {10, 20, 30, 40, 50, 60, 70, 80} sobre un 10% del dataset usando silhouette score con distancia coseno. El resultado se cachea en `models/best_k.json`.
2. **Entrenamiento final**: entrena con el K optimo sobre el dataset completo.
3. **Artefactos guardados**:
   - `models/kmeans_spark/` — modelo Spark ML
   - `models/cluster_assignments.parquet` — cluster y distancia al centroide por restaurante
   - `models/feature_cols.json` — columnas de features usadas
   - `models/k_selection.png` — grafica silhouette score vs K

---

## Streaming (`streaming_score.py`)

Consumidor de Spark Structured Streaming que clasifica en tiempo real los restaurantes que llegan por Kafka:

1. Lee el schema del topic `restaurants_schema`
2. Consume el topic `restaurants` en micro-batches de 5 segundos
3. Por cada batch: ensambla el vector de features y aplica el modelo KMeans
4. Calcula la distancia euclidea al centroide del cluster asignado
5. Escribe `row_id, nombre, cluster, distancia, timestamp` en `data/streaming/results.csv`

La app Streamlit lee ese CSV cada 5 segundos para actualizar el tab "En vivo".

---

## Aplicacion Streamlit (`app.py`)

### Tab Explorar
- Filtra el millon de restaurantes por pais, ciudad, precio, cocina y dieta
- Busqueda por nombre de restaurante
- Estadisticas en tiempo real del subconjunto filtrado
- Click en un restaurante → 10 recomendaciones del mismo cluster ordenadas por distancia al centroide

### Tab En vivo
- Muestra unicamente los restaurantes que han llegado por el stream (los ya clasificados por Spark)
- Mismos filtros que Explorar, pero acotados al subconjunto scoreado
- Auto-refresh cada 5 segundos via `@st.fragment`
- Grafico de distribucion por cluster
- Recomendaciones calculadas solo sobre restaurantes del stream

---

## Configuracion

`config.toml` centraliza los parametros del pipeline:

| Seccion | Clave | Descripcion |
|---|---|---|
| `[general]` | `chunk_size` | Filas leidas por batch |
| `[clean]` | `null_threshold` | Fraccion de nulos para eliminar una columna |
| `[clean]` | `numeric_categorical_threshold` | Valores unicos maximos para clasificar como categorica |
| `[eda]` | `top_n_categories` | Categorias mostradas en graficos de barras |
| `[preprocessing]` | `ohe_max_cardinality` | Cardinalidad maxima para OHE |
| `[kafka]` | `bootstrap_servers`, `topic` | Conexion y topic del productor Kafka |

---

