# Consumidor de Spark Structured Streaming que indexa los restaurantes
# que llegan por Kafka en tiempo real para búsqueda por filtros.
#
# A diferencia de streaming_score.py, este consumidor NO hace inferencia KMeans.
# Su único objetivo es escribir los registros parseados lo más rápido posible
# en data/streaming/filter_results.csv para que main.py los sirva en el tab
# de búsqueda sin depender del ritmo del consumidor KMeans.
#
#   1. Lee el schema del topic 'restaurants_schema' (enviado por producer.py)
#   2. Consume el topic 'restaurants' con maxOffsetsPerTrigger alto (procesado rápido)
#   3. Por cada micro-batch: escribe los registros parseados en filter_results.csv
#      (main.py lo lee para el endpoint /api/restaurants)

import json
import os
import orjson
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    StructType, StructField,
    FloatType, DoubleType, IntegerType, LongType, StringType,
)

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"

BASE_DIR = Path(__file__).parent
FEATURE_COLS_JSON = BASE_DIR / "models" / "feature_cols.json"
FILTER_OUT = BASE_DIR / "data" / "streaming" / "filter_results.csv"

TOPIC = "restaurants"
BOOTSTRAP = "localhost:9092"

# Offsets por trigger más altos que en streaming_score.py para ir más rápido.
# El consumidor de filtros no hace ML, así que puede procesar batches grandes.
MAX_OFFSETS_PER_TRIGGER = 5000


def sanitize_col(name: str) -> str:
    return name.replace(".", "_")


def map_spark_types(type_dict: dict) -> StructType:
    mapping = {
        "numeric":       DoubleType(),
        "ohe":           IntegerType(),
        "label_encoded": LongType(),
        "list_json":     StringType(),
    }
    return StructType([
        StructField(name, mapping.get(t, StringType()), True)
        for name, t in type_dict.items()
    ])


def main():
    spark = (SparkSession.builder
             .appName("RestaurantFilterConsumer")
             .master("local[*]")
             .config("spark.jars.packages",
                     "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1")
             .config("spark.driver.memory", "2g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # Leer schema desde el topic de metadatos — igual que streaming_score.py
    print(f"Leyendo schema del topic '{TOPIC}_schema' ...")
    raw_schema = (spark.read
                  .format("kafka")
                  .option("kafka.bootstrap.servers", BOOTSTRAP)
                  .option("subscribe", f"{TOPIC}_schema")
                  .option("startingOffsets", "earliest")
                  .load()
                  .selectExpr("CAST(value AS STRING)", "offset")
                  .orderBy("offset", ascending=False)
                  .limit(1)
                  .collect())

    if not raw_schema:
        print("ERROR: no se encontró schema en 'restaurants_schema'. "
              "Ejecuta tasks/producer.py primero.")
        spark.stop()
        return

    type_dict = orjson.loads(raw_schema[0][0])
    type_dict["row_id"] = "label_encoded"
    spark_schema = map_spark_types(type_dict)
    print(f"  Schema cargado: {len(type_dict)} columnas")

    # Stream de Kafka con offsets altos para máximo throughput
    raw_stream = (spark.readStream
                  .format("kafka")
                  .option("kafka.bootstrap.servers", BOOTSTRAP)
                  .option("subscribe", TOPIC)
                  .option("startingOffsets", "earliest")
                  .option("maxOffsetsPerTrigger", MAX_OFFSETS_PER_TRIGGER)
                  .load())

    parsed = (raw_stream
              .selectExpr("CAST(value AS STRING) as json_payload",
                          "timestamp as kafka_ts")
              .select(from_json(col("json_payload"), spark_schema).alias("d"),
                      col("kafka_ts"))
              .select("d.*", "kafka_ts"))

    # Renombrar columnas con '.' igual que en streaming_score.py
    rename_map = {c: sanitize_col(c) for c in type_dict if "." in c}
    if rename_map:
        parsed = parsed.select(
            [col(f"`{c}`").alias(sanitize_col(c)) if "." in c else col(c)
             for c in parsed.columns]
        )

    FILTER_OUT.parent.mkdir(parents=True, exist_ok=True)
    write_header = [not FILTER_OUT.exists()]

    def write_batch(batch_df, epoch_id):
        if batch_df.isEmpty():
            return

        # Solo guardamos row_id y los campos que usa main.py para filtrar/mostrar.
        # Guardamos también kafka_ts para que el snapshot pueda ordenar por llegada.
        keep_cols = ["row_id", "restaurant_name", "country", "city",
                     "avg_rating", "price_level",
                     "vegetarian_friendly", "vegan_options", "gluten_free",
                     "kafka_ts"]

        # Seleccionamos solo las columnas que existen en el batch
        available = [c for c in keep_cols if c in batch_df.columns]
        pdf = batch_df.select(available).toPandas()

        print(f"\n[FilterConsumer] Batch {epoch_id} — {len(pdf)} registros")

        pdf.to_csv(FILTER_OUT,
                   mode="a",
                   header=write_header[0],
                   index=False)
        write_header[0] = False

    query = (parsed.writeStream
             .outputMode("append")
             .foreachBatch(write_batch)
             .trigger(processingTime="2 seconds")   # trigger más frecuente que el scorer
             .start())

    print(f"\nFilter consumer arrancado.")
    print(f"  Topic   : {TOPIC}  ({BOOTSTRAP})")
    print(f"  Salida  : {FILTER_OUT}")
    print("  Ctrl+C para parar.\n")

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        query.stop()
        spark.stop()
        print("\nParado.")


if __name__ == "__main__":
    main()
