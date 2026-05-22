import logging
import tomllib
import polars as pl
from pathlib import Path

# Rutas de entrada y salida
SOURCE_CSV = Path(__file__).parent.parent / "tripadvisor_european_restaurants.csv"
OUTPUT_CSV = Path(__file__).parent.parent / "data" / "raw" / "raw.csv"

with open(Path(__file__).parent.parent / "config.toml", "rb") as _f:
    _config = tomllib.load(_f)

CHUNK_SIZE = _config["general"]["chunk_size"]

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
    for i, chunk in enumerate(pl.scan_csv(SOURCE_CSV).collect_batches(chunk_size=CHUNK_SIZE)):
        # El primer chunk escribe el header, añadimos los siguientes sin header
        if first_chunk:
            chunk.write_csv(OUTPUT_CSV)
            first_chunk = False
        else:
            with open(OUTPUT_CSV, "a") as f:
                f.write(chunk.write_csv(include_header=False))
        total_rows += len(chunk)
        log.info("Batch %d — %d filas acumuladas", i + 1, total_rows)

    log.info("Extraccion completa: %d filas guardadas en %s", total_rows, OUTPUT_CSV)

if __name__ == "__main__":
    extract()
