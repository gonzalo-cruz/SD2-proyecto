import orjson
import logging
import os
import sys
import tomllib
import pandas as pd
from pathlib import Path
from confluent_kafka import Producer

# Configuración de rutas
BASE_DIR = Path(__file__).parent.parent
CONFIG_PATH = BASE_DIR / "config.toml"
INPUT_CSV = BASE_DIR / "data" / "processed" / "preprocessed.csv"
TYPE_DICT_PATH = BASE_DIR / "data" / "processed" / "type_dict_encoding.json"

# Carga de configuración
try:
    with open(CONFIG_PATH, "rb") as _f:
        _config = tomllib.load(_f).get("kafka", {})
except FileNotFoundError:
    _config = {}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

total_sent = 0
total_errors = 0

def delivery_report(err, msg):
    global total_sent, total_errors
    if err is not None:
        log.error(f"Error: {err}")
        total_errors += 1
    else:
        total_sent += 1

def load():
    kafka_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", _config.get("bootstrap_servers", "localhost:9092"))
    topic = os.environ.get("KAFKA_TOPIC", _config.get("topic", "restaurants"))
    
    producer = Producer({'bootstrap.servers': kafka_servers, 'acks': 'all'})

    # Cargar y filtrar diccionario de tipos (excluyendo _json)
    with open(TYPE_DICT_PATH, "rb") as f:
        full_types = orjson.loads(f.read())
    
    filtered_schema = {k: v for k, v in full_types.items() if v != "list_json"}

    # Enviar esquema al topic de metadatos
    log.info("Enviando esquema a Kafka...")
    producer.produce(
        topic=f"{topic}_schema",
        value=orjson.dumps(filtered_schema),
        callback=delivery_report
    )
    producer.flush()

    # Enviar datos del CSV
    log.info(f"Iniciando envío de datos desde {INPUT_CSV}")
    row_id = 0
    for chunk in pd.read_csv(INPUT_CSV, chunksize=10000, low_memory=False):
        for record in chunk.to_dict(orient="records"):
            record["row_id"] = row_id
            row_id += 1
            # Serialización ultra rápida con orjson
            value_bytes = orjson.dumps(record)
            
            while True:
                try:
                    producer.produce(topic=topic, value=value_bytes, callback=delivery_report)
                    break
                except BufferError:
                    producer.poll(1.0)
            producer.poll(0)

    producer.flush()
    log.info(f"Finalizado. Enviados: {total_sent} | Errores: {total_errors}")

if __name__ == "__main__":
    load()