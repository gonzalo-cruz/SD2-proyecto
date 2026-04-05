# TripAdvisor Restaurants Pipeline

Pipeline de datos con Apache Airflow para procesar el dataset de restaurantes europeos de TripAdvisor (~1M filas, 42 columnas).

---

## Estructura

```
proyecto/
├── dags/
│   └── pipeline.py        
├── tasks/
│   ├── extract.py
│   ├── clean.py
│   ├── eda.py
│   ├── preprocessing.py
│   └── load.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── eda/
└── pyproject.toml
```

---

## Cómo ejecutar

### 1. Instalar dependencias

```bash
cd sd2/proyecto
uv sync
```

### 2. Levantar Airflow

```bash
AIRFLOW_HOME=$(pwd) uv run airflow standalone
```

La UI queda en **http://localhost:8080**. Las credenciales se generan en `simple_auth_manager_passwords.json.generated`.

### 3. Lanzar el pipeline

Desde la UI: activar el toggle del DAG `tripadvisor_pipeline` → **Trigger DAG**.

O desde terminal:

```bash
AIRFLOW_HOME=$(pwd) uv run airflow dags trigger tripadvisor_pipeline
```

### 4. Ejecutar una task individualmente (para probar)

```bash
uv run python -m tasks.extract
uv run python -m tasks.clean
uv run python -m tasks.eda
uv run python -m tasks.preprocessing
```

> **Nota**: la task `load` requiere un broker de Kafka corriendo (ver sección [Kafka](#kafka) más abajo).

### 5. Levantar Kafka (necesario para la task `load`)

```bash
docker run -d --name kafka -p 9092:9092 apache/kafka:latest
```

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


### preprocessing
- Normalización con `StandardScaler`
- One-Hot Encoding para columnas categóricas de baja cardinalidad (≤ 15 valores únicos)
- PCA incremental sobre columnas numéricas
- Todo se procesa en batches para no cargar el dataset entero en memoria
- Guarda `preprocessed.csv`, `pca.csv`, `scaler.pkl`, `pca.pkl` en `data/processed/`

### load

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
