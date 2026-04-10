import json
import logging
import os
import tomllib
import pandas as pd
from pathlib import Path
from kafka import KafkaProducer
from kafka.errors import KafkaError

CONFIG_PATH = Path(__file__).parent.parent / "config.toml"
with open(CONFIG_PATH, "rb") as _f:
    _config = tomllib.load(_f).get("kafka", {})

# Ruta del dataset preprocesado
INPUT_CSV = Path(__file__).parent.parent / "data" / "processed" / "preprocessed.csv"

CHUNK_SIZE = 50_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load():
    # Configuración leída de variables de entorno (con valores por defecto)
    kafka_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", _config.get("bootstrap_servers", "localhost:9092"))
    topic = os.environ.get("KAFKA_TOPIC", _config.get("topic", "restaurants"))
    batch_size = int(os.environ.get("KAFKA_BATCH_SIZE", _config.get("batch_size", 500)))

    log.info("Conectando a Kafka en %s, topic: %s", kafka_servers, topic)

    # El serializer convierte cada mensaje (dict) a JSON en bytes
    producer = KafkaProducer(
        bootstrap_servers=kafka_servers,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        acks="all",   # esperar confirmación de todos los brokers
        retries=3,
    )

    total_sent = 0
    total_errors = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE, low_memory=False)):
        # Convertir el chunk a una lista de diccionarios (una fila = un mensaje)
        records = chunk.to_dict(orient="records")

        # Enviar en sub-batches para no gastar demasiada memoria
        for start in range(0, len(records), batch_size):
            batch   = records[start: start + batch_size]
            futures = [producer.send(topic, value=record) for record in batch]

            # Esperar confirmación de cada mensaje
            for future in futures:
                try:
                    future.get(timeout=10)
                    total_sent += 1
                except KafkaError as e:
                    log.error("Error enviando mensaje: %s", e)
                    total_errors += 1

        log.info("Chunk %d enviado — total: %d mensajes, errores: %d",
                 chunk_idx + 1, total_sent, total_errors)

    producer.flush()
    producer.close()
    log.info("Carga finalizada — enviados: %d | errores: %d", total_sent, total_errors)

if __name__ == "__main__":
    load()
