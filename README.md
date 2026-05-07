# **TripAdvisor Restaurants Pipeline**

Sistema integral de procesamiento de datos para el dataset de restaurantes europeos de TripAdvisor (\~1M filas, 42 columnas). El proyecto se divide en un pipeline ETL batch orquestado por Airflow y una fase de procesamiento de eventos en tiempo real con Kafka y Spark Streaming.

## **Estructura del Proyecto (ETL \+ Streaming)**

proyecto/  
├── dags/  
│   └── pipeline.py                  \# Definición del DAG de Airflow  
├── tasks/  
│   ├── extract.py                   \# Descarga/Lectura de datos raw  
│   ├── clean.py                     \# Limpieza y tipado inicial  
│   ├── eda.py                       \# Análisis exploratorio y generación de gráficas  
│   ├── preprocessing.py             \# Feature engineering (OHE, Escalado, PCA)  
│   ├── load.py                      \# Orquestador del productor (BashOperator)  
│   └── producer.py                  \# Productor Kafka (confluent\_kafka \+ orjson)  
├── data/  
│   ├── raw/  
│   │   └── raw.csv                  \# Dataset original  
│   └── processed/  
│       ├── clean.csv                \# Datos limpios  
│       ├── type\_dict\_encoding.json  \# Esquema técnico para Spark  
│       ├── encodings.json           \# Diccionarios de decodificación  
│       ├── preprocessed.csv         \# Dataset final con OHE y transformaciones  
│       └── ...                      \# Metadatos y modelos (.json, .pkl)  
├── docker-compose.yml               \# Infraestructura de Kafka (KRaft)  
├── pyproject.toml                   \# Gestión de dependencias con uv  
├── config.toml                      \# Configuración centralizada del pipeline  
├── salida\_stream.txt                \# Log del consumidor principal  
└── salida\[1-6\].txt                  \# Resultados de las consultas analíticas

## **Requisitos**

* Python 3.11+  
* [uv](https://github.com/astral-sh/uv) — gestiona las dependencias (incluye Airflow)  
* Docker (para Kafka)  
* Java 21 en /usr/lib/jvm/java-21-openjdk-amd64 — ruta fijada en queries.py y struct\_kafka\_consumer\_local.py; si la instalación es diferente, ajustar la variable JAVA\_HOME en esos ficheros

# **Parte 1: Pipeline ETL (Batch)**

Esta fase procesa el dataset desde su estado bruto hasta un formato listo para el análisis o entrenamiento de modelos.

## **Ejecución Manual del ETL**

Aunque el flujo está diseñado para Airflow, los scripts pueden ejecutarse secuencialmente mediante uv:

1. **Extracción:** uv run tasks/extract.py  
2. **Limpieza:** uv run tasks/clean.py  
3. **Análisis (EDA):** uv run tasks/eda.py  
4. **Preprocesamiento:** uv run tasks/preprocessing.py  
5. **Carga (Productor):** uv run tasks/load.py (Inicia el envío a Kafka)

## **Orquestación con Airflow**

El pipeline se gestiona automáticamente mediante el DAG definido en dags/pipeline.py. A continuación se detallan los pasos necesarios para configurar el entorno e iniciar la orquestación:

### **1\. Descargar el dataset**

Descarga el dataset desde [Kaggle](https://www.kaggle.com/datasets/stefanoleone992/tripadvisor-european-restaurants/data) y coloca el fichero tripadvisor\_european\_restaurants.csv en la raíz del proyecto. La task extract lo cogerá de ahí y generará data/raw/raw.csv automáticamente.

### **2\. Instalar dependencias**

cd sd2/proyecto  
uv sync

### **3\. Levantar Airflow**

export AIRFLOW\_HOME=$(pwd)  
uv run airflow standalone

**Nota:** Es fundamental mantener las rutas relativas de las tasks para que el DAG pueda importar y ejecutar los scripts de la carpeta tasks/ correctamente una vez se inicie el scheduler.

# **Parte 2: Introducción al Streaming (Spark & Kafka)**

Esta sección se enfoca en la ingesta y el procesamiento de flujos de datos en tiempo real una vez que el ETL ha generado los metadatos necesarios.

## **Ficheros Relevantes**

* struct\_kafka\_consumer\_local.py (Consumidor principal / Unbounded Table)  
* queries.py (Validación de consultas de negocio)  
* producer.py (Ingesta desde CSV preprocesado a Kafka)  
* type\_dict\_encoding.json y encodings.json (Contratos de datos)

## **Instrucciones de Ejecución**

Siga estrictamente este orden para asegurar la conectividad:

1. **Sincronizar el Entorno:**  
   uv sync

2. **Levantar Infraestructura:**  
   docker compose up \-d

3. **Iniciar Productor:**  
   uv run tasks/producer.py

4. **Ejecutar Consumidores (a elegir):**  
   * **Tabla Dinámica:** uv run struct\_kafka\_consumer\_local.py  
   * **Consultas Analíticas:** uv run queries.py

## **Notas Finales**

* **Entorno:** uv sync garantiza que se utilice **PySpark 4.1.1** y las versiones exactas de las librerías de Kafka.  
* **Limpieza:** Use docker compose down \-v para resetear el broker y eliminar los tópicos creados.