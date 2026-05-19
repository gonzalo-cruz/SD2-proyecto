# Rutas y constantes globales usadas por toda la aplicación
from pathlib import Path

# Directorio raíz del proyecto (donde está este fichero)
BASE_DIR       = Path(__file__).parent

# Datos procesados por el pipeline ETL de Airflow
CLEAN_CSV      = BASE_DIR / "data" / "processed" / "clean.csv"
ENCODINGS_JSON = BASE_DIR / "data" / "processed" / "encodings.json"
CUISINES_JSON  = BASE_DIR / "data" / "processed" / "cuisines.json"

# Resultado del entrenamiento KMeans en Spark (cluster asignado a cada restaurante)
ASSIGNMENTS_PQ = BASE_DIR / "models" / "cluster_assignments.parquet"

# CSV que va escribiendo streaming_score.py con los restaurantes llegados por Kafka
STREAM_CSV     = BASE_DIR / "data" / "streaming" / "results.csv"

# Máximo de filas que mostramos en la tabla para no saturar el navegador
DISPLAY_LIMIT = 200
