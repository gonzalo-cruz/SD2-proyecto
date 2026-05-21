# TripAdvisor Restaurants Pipeline

Pipeline de datos con Apache Airflow para procesar el dataset de restaurantes europeos de TripAdvisor (~1M filas, 42 columnas), entrenar un modelo KMeans con Spark ML y exponerlo en una aplicación web con soporte de streaming en tiempo real via Kafka.

---

## Estructura

```
proyecto/
├── streaming_score.py               # Consumidor Spark Structured Streaming (KMeans)
├── streaming_filter.py              # Consumidor Spark Structured Streaming (filtrado rápido)
├── train_model.py                   # Entrenamiento KMeans con Spark ML
├── run.sh                           # Script de arranque completo (infraestructura + servicios)
│
├── webapp/                          # Aplicación web (FastAPI + React)
│   ├── backend/
│   │   └── main.py                  #   API REST: filtros, recomendaciones, SSE, cola de prioridad
│   └── frontend/
│       ├── src/
│       │   ├── App.jsx              #   Raíz: tabs Explorar / En vivo
│       │   ├── ExploreTab.jsx       #   Búsqueda y recomendaciones
│       │   ├── LiveTab.jsx          #   Stream en tiempo real con SSE
│       │   └── RecsPanel.jsx        #   Panel de recomendaciones
│       ├── package.json
│       └── vite.config.js
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
│   ├── informe.md                   #   Descripcion del grafo del pipeline
│   └── grafo_pipeline.png           #   Imagen del grafo de Airflow
│
├── data/                            # Datos generados por el pipeline (no en git)
│   ├── raw/                         #   raw.csv generado por extract
│   ├── processed/                   #   clean.csv, preprocessed.csv y artefactos ETL
│   └── streaming/                   #   results.csv y filter_results.csv generados por los consumidores
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
- Java 21
- Node.js 22

---

## Instalacion

```bash
git clone <repo> && cd proyecto
uv sync
cd webapp/frontend && npm install
```

---

## Ejecucion completa

### 1. Dataset

Descarga `tripadvisor_european_restaurants.csv` desde [Kaggle](https://www.kaggle.com/datasets/stefanoleone992/tripadvisor-european-restaurants/data) y colócalo en la raiz del proyecto.

### 2. Instalación de requisitos

Bibliotecas de Python:
```bash
uv sync
source .venv/bin/activate
```

OpenJDK:
```bash
sudo apt install openjdk-21-jdk
```

Node.js:
```bash
sudo apt install nvm
nvm install 22
nvm use 22
```

Bibliotecas de Node.js:
```bash
cd webapp/frontend
npm install
```

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
uv run python -m tasks.load
```

### 3. Entrenar el modelo KMeans

```bash
uv run train_model.py
```

Genera `models/kmeans_spark/` y `models/cluster_assignments.parquet`.

### 4. Lanzar la aplicacion de streaming completa

```bash
bash run.sh
```

El script arranca automáticamente: Docker (Kafka), consumidores Spark, backend FastAPI y frontend React. También maneja el cierre correcto de estos procesos.
La app queda disponible en **http://localhost:5173**.

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

## Streaming

### `streaming_filter.py` — Consumidor rápido (sin ML)
Consume el topic `restaurants` con alta frecuencia y escribe los registros parseados en `data/streaming/filter_results.csv`. Permite que la pestaña Explorar muestre resultados antes de que KMeans los clasifique.

### `streaming_score.py` — Consumidor KMeans
1. Lee el schema del topic `restaurants_schema`
2. Consume en paralelo el topic `restaurants` (normal) y `restaurants_priority` (selecciones del usuario)
3. Por cada batch: ensambla el vector de features y aplica el modelo KMeans
4. Calcula la distancia euclidea al centroide del cluster asignado
5. Escribe `row_id, nombre, cluster, distancia, timestamp` en `data/streaming/results.csv`

---

## Aplicacion web (`webapp/`)

### Backend (FastAPI — puerto 8000)
- `GET /api/restaurants` — lista filtrable de restaurantes (usa `filter_results.csv`)
- `GET /api/recommendations/{row_id}` — 10 recomendaciones por similitud coseno dentro del cluster
- `POST /api/priority/{row_id}` — encola un restaurante en `restaurants_priority` para clasificacion inmediata
- `GET /api/stream/snapshot` — snapshot filtrado de los restaurantes ya scoreados por KMeans
- `GET /api/stream/events` — Server-Sent Events con nuevos registros scoreados en tiempo real
- `GET /api/stream/stats` — total scoreados y distribucion por cluster
- `GET /api/filter/stats` — total procesados por el consumidor de filtros

### Frontend (React + Vite — puerto 5173)
- **Explorar**: filtros por país, ciudad, precio, rating y dieta; búsqueda por nombre; recomendaciones al seleccionar un restaurante
- **En vivo**: tabla actualizada en tiempo real via SSE, contadores de procesados/scoreados, gráfico de distribución por cluster

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
