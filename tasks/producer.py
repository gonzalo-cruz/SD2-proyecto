import json
import logging
import os
import sys
import tomllib
import pandas as pd
from pathlib import Path
from confluent_kafka import Producer

# Configuración de rutas
CONFIG_PATH = Path(__file__).parent.parent / "config.toml"
try:
    with open(CONFIG_PATH, "rb") as _f:
        _config = tomllib.load(_f).get("kafka", {})
except FileNotFoundError:
    _config = {}

# Ruta del dataset preprocesado
INPUT_CSV = Path(__file__).parent.parent / "data" / "processed" / "preprocessed.csv"
CHUNK_SIZE = 50_000

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Contadores globales para el callback de entrega
total_sent = 0
total_errors = 0

def delivery_report(err, msg):
    """
    Se llama una vez por cada mensaje producido para indicar el resultado de la entrega.
    Se activa al llamar a poll() o flush().
    """
    global total_sent, total_errors
    if err is not None:
        log.error("Error enviando mensaje: %s", err)
        total_errors += 1
    else:
        total_sent += 1

def load():
    # Configuración de variables de entorno
    kafka_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", _config.get("bootstrap_servers", "localhost:9092"))
    topic = os.environ.get("KAFKA_TOPIC", _config.get("topic", "restaurants"))
    batch_size = int(os.environ.get("KAFKA_BATCH_SIZE", _config.get("batch_size", 500)))

    log.info("Conectando a Kafka en %s, topic: %s", kafka_servers, topic)

    # Configuración de confluent_kafka
    conf = {
        'bootstrap.servers': kafka_servers,
        'acks': 'all',
        'retries': 3,
        # confluent_kafka maneja el batching internamente basándose en este parámetro
        'batch.num.messages': batch_size 
    }

    try:
        producer = Producer(conf)

        for chunk_idx, chunk in enumerate(pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE, low_memory=False)):
            records = chunk.to_dict(orient="records")

            for record in records:
                # Serializar manualmente el diccionario a una cadena de bytes JSON
                value_bytes = json.dumps(record, default=str).encode("utf-8")

                # Producir el mensaje. Manejar BufferError si la cola interna se llena
                while True:
                    try:
                        producer.produce(
                            topic=topic, 
                            value=value_bytes, 
                            callback=delivery_report
                        )
                        break
                    except BufferError:
                        # La cola está llena, esperar 1 segundo para liberar espacio
                        producer.poll(1.0)
                
                # Atender la cola de callbacks de entrega de forma asíncrona
                producer.poll(0)

            log.info("Chunk %d procesado en cola...", chunk_idx + 1)

        # Esperar a que se entreguen los mensajes pendientes
        log.info("Haciendo flush final a Kafka. Esperando confirmaciones...")
        producer.flush()
        
        log.info("Carga finalizada — enviados: %d | errores: %d", total_sent, total_errors)

    except Exception as e:
        # Esto captura errores fatales y fuerza un código de salida de error para Airflow
        log.critical("Error fatal durante la ejecución del script: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    load()