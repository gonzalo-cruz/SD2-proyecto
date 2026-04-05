import logging
import pandas as pd
from pathlib import Path

# Rutas de entrada y salida
SOURCE_CSV = Path(__file__).parent.parent / "tripadvisor_european_restaurants.csv"
OUTPUT_CSV = Path(__file__).parent.parent / "data" / "raw" / "raw.csv"

CHUNK_SIZE = 50_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def extract():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    first_chunk = True

    log.info("Iniciando extracción desde %s", SOURCE_CSV)

    # Leemos el CSV en trozos para no cargar todo en memoria
    for i, chunk in enumerate(pd.read_csv(SOURCE_CSV, chunksize=CHUNK_SIZE, low_memory=False)):
        # El primer chunk escribe el header, añadimos los siguientes sin header
        chunk.to_csv(OUTPUT_CSV, mode="w" if first_chunk else "a", header=first_chunk, index=False)
        first_chunk = False
        total_rows += len(chunk)
        log.info("Batch %d — %d filas acumuladas", i + 1, total_rows)

    log.info("Extraccion completa: %d filas guardadas en %s", total_rows, OUTPUT_CSV)

if __name__ == "__main__":
    extract()
