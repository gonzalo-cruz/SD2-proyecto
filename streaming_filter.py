# Consumidor de Spark Structured Streaming que actúa como tracker de offsets
# para la búsqueda por filtros en tiempo real.
#
# A diferencia de streaming_score.py, este consumidor NO hace inferencia KMeans
# ni necesita cargar ningún dataset en memoria.
# Su único objetivo es extraer el row_id de cada mensaje Kafka y escribirlo
# en data/streaming/filter_results.csv lo más rápido posible.
#
# main.py lee esa lista de row_ids y la usa como máscara sobre df (clean.csv,
# ya cargado en memoria) para mostrar únicamente los restaurantes que el stream
# ha alcanzado, con todos sus campos correctamente decodificados.
#
#   1. Consume el topic 'restaurants' con maxOffsetsPerTrigger alto
#   2. Por cada micro-batch: extrae row_id y lo escribe en filter_results.csv

import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, get_json_object

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"

BASE_DIR   = Path(__file__).parent
FILTER_OUT = BASE_DIR / "data" / "streaming" / "filter_results.csv"

TOPIC     = "restaurants"
BOOTSTRAP = "localhost:9092"

# Offsets por trigger altos: sin ML, podemos procesar batches muy grandes.
MAX_OFFSETS_PER_TRIGGER = 5000


def main():
    spark = (SparkSession.builder
             .appName("RestaurantFilterConsumer")
             .master("local[*]")
             .config("spark.jars.packages",
                     "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1")
             .config("spark.driver.memory", "1g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # No necesitamos leer el schema del topic de metadatos: solo extraemos
    # row_id del JSON crudo con get_json_object, sin parsear el mensaje entero.
    raw_stream = (spark.readStream
                  .format("kafka")
                  .option("kafka.bootstrap.servers", BOOTSTRAP)
                  .option("subscribe", TOPIC)
                  .option("startingOffsets", "earliest")
                  .option("maxOffsetsPerTrigger", MAX_OFFSETS_PER_TRIGGER)
                  .load())

    # Extraemos únicamente row_id del JSON — sin schema completo, sin OHE
    parsed = (raw_stream
              .selectExpr("CAST(value AS STRING) as json_payload")
              .select(get_json_object(col("json_payload"), "$.row_id").cast("long").alias("row_id")))

    FILTER_OUT.parent.mkdir(parents=True, exist_ok=True)
    write_header = [not FILTER_OUT.exists()]

    def write_batch(batch_df, epoch_id):
        if batch_df.isEmpty():
            return

        pdf = batch_df.dropna(subset=["row_id"]).toPandas()
        if pdf.empty:
            return

        print(f"\n[FilterConsumer] Batch {epoch_id} — {len(pdf)} row_ids")

        pdf.to_csv(FILTER_OUT,
                   mode="a",
                   header=write_header[0],
                   index=False)
        write_header[0] = False

    query = (parsed.writeStream
             .outputMode("append")
             .foreachBatch(write_batch)
             .trigger(processingTime="2 seconds")
             .start())

    print(f"\nFilter consumer arrancado.")
    print(f"  Topic  : {TOPIC}  ({BOOTSTRAP})")
    print(f"  Salida : {FILTER_OUT}")
    print("  Ctrl+C para parar.\n")

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        query.stop()
        spark.stop()
        print("\nParado.")


if __name__ == "__main__":
    main()
