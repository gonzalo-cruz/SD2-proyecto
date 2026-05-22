# Consumidor de Spark Structured Streaming que clasifica los restaurantes
# que llegan por Kafka en tiempo real usando el modelo KMeans entrenado.
#   - Lee de DOS topics en paralelo:
#       · 'restaurants'          → stream normal (mismo ritmo que antes)
#       · 'restaurants_priority' → cola de prioridad para restaurantes que el
#                                  usuario ha seleccionado antes de que el
#                                  consumidor KMeans haya llegado a ellos
#   - Mantiene un conjunto 'seen_row_ids' para no clasificar dos veces el mismo
#     registro: cuando un restaurante llega por la cola de prioridad se clasifica
#     inmediatamente y su row_id se añade al conjunto; cuando ese mismo registro
#     llega por el stream normal se descarta.
#   - El conjunto seen_row_ids se persiste en disco (seen_row_ids.json) para
#     sobrevivir reinicios del proceso.
#
#   Flujo original:
#   1. Lee el schema del topic 'restaurants_schema'
#   2. Consume micro-batches de 5 s del topic normal
#   3. Ensambla features → KMeans → distancia al centroide
#   4. Escribe resultados en data/streaming/results.csv

import atexit
import json
import os
import numpy as np
import orjson
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, lit
from pyspark.sql.types import (
    StructType, StructField,
    FloatType, DoubleType, IntegerType, LongType, StringType,
)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeansModel

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"

BASE_DIR = Path(__file__).parent
ENCODINGS_JSON = BASE_DIR / "data" / "processed" / "encodings.json"
FEATURE_COLS_JSON = BASE_DIR / "models" / "feature_cols.json"
KMEANS_MODEL = BASE_DIR / "models" / "kmeans_spark"
STREAM_OUT = BASE_DIR / "data" / "streaming" / "results.csv"
SEEN_IDS_PATH = BASE_DIR / "data" / "streaming" / "seen_row_ids.json"

TOPIC = "restaurants"
PRIORITY_TOPIC = "restaurants_priority"
BOOTSTRAP = "localhost:9092"


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


def load_seen_ids() -> set:
    """Carga el conjunto de row_ids ya clasificados desde disco (para sobrevivir reinicios)."""
    if SEEN_IDS_PATH.exists():
        try:
            with open(SEEN_IDS_PATH) as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()


def save_seen_ids(seen: set):
    """Persiste el conjunto de row_ids en disco."""
    SEEN_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SEEN_IDS_PATH, "w") as f:
        json.dump(list(seen), f)


def main():
    with open(FEATURE_COLS_JSON) as f:
        feature_cols = json.load(f)

    with open(ENCODINGS_JSON) as f:
        enc = json.load(f)
    decode_name = {int(v): k for k, v in enc["restaurant_name"].items()}

    # Conjunto de row_ids ya clasificados (en memoria + persistido en disco)
    seen_row_ids: set = load_seen_ids()
    atexit.register(save_seen_ids, seen_row_ids)

    spark = (SparkSession.builder
             .appName("RestaurantStreamScorer")
             .master("local[*]")
             .config("spark.jars.packages",
                     "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1")
             .config("spark.driver.memory", "2g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # Leer schema del topic de metadatos
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

    model = KMeansModel.load(str(KMEANS_MODEL))
    centers = np.array(model.clusterCenters(), dtype=np.float32)
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features",
                                handleInvalid="keep")

    def build_stream(topic: str, source_label: str, max_offsets: int):
        """Construye un readStream de Kafka y añade una columna 'source' para identificar el topic."""
        raw = (spark.readStream
               .format("kafka")
               .option("kafka.bootstrap.servers", BOOTSTRAP)
               .option("subscribe", topic)
               .option("startingOffsets", "earliest")
               .option("maxOffsetsPerTrigger", max_offsets)
               .load())

        parsed = (raw
                  .selectExpr("CAST(value AS STRING) as json_payload",
                              "timestamp as kafka_ts")
                  .select(from_json(col("json_payload"), spark_schema).alias("d"),
                          col("kafka_ts"))
                  .select("d.*", "kafka_ts")
                  .withColumn("_source", lit(source_label)))

        rename_map = {c: sanitize_col(c) for c in type_dict if "." in c}
        if rename_map:
            parsed = parsed.select(
                [col(f"`{c}`").alias(sanitize_col(c)) if "." in c else col(c)
                 for c in parsed.columns]
            )
        return parsed

    # Stream normal — mismo ritmo que antes
    normal_stream = build_stream(TOPIC, "normal", max_offsets=500)

    # Stream de prioridad — offsets altos para vaciar la cola rápido.
    # Al ser un topic de prioridad se espera tráfico bajo y esporádico.
    priority_stream = build_stream(PRIORITY_TOPIC, "priority", max_offsets=5000)

    # Unión de ambos streams: Spark los procesa en el mismo micro-batch
    combined_stream = normal_stream.union(priority_stream)

    STREAM_OUT.parent.mkdir(parents=True, exist_ok=True)
    write_header = [not STREAM_OUT.exists()]

    def score_batch(batch_df, epoch_id):
        nonlocal seen_row_ids

        if batch_df.isEmpty():
            return

        pdf = batch_df.toPandas()

        # Separar registros por origen 
        priority_pdf = pdf[pdf["_source"] == "priority"].copy()
        normal_pdf   = pdf[pdf["_source"] == "normal"].copy()

        # Los registros de prioridad se procesan siempre y se marcan como vistos
        new_priority_ids = set(priority_pdf["row_id"].dropna().astype(int).tolist())

        # Los registros normales se filtran: descartamos los ya clasificados
        normal_pdf = normal_pdf[
            ~normal_pdf["row_id"].astype(int).isin(seen_row_ids)
        ]

        # Unimos los que hay que clasificar en este batch
        to_score = pd.concat([priority_pdf, normal_pdf], ignore_index=True)
        if to_score.empty:
            return

        # Convertimos de vuelta a Spark para usar el pipeline de ML
        to_score_spark = spark.createDataFrame(to_score)
        featured = assembler.transform(to_score_spark)
        scored   = model.transform(featured)

        result_pdf = scored.select(
            "row_id", "restaurant_name", "kafka_ts", "cluster", "features", "_source"
        ).toPandas()

        feat_mat    = np.array(result_pdf["features"].tolist(), dtype=np.float32)
        cluster_ids = result_pdf["cluster"].values.astype(int)
        dists       = np.linalg.norm(feat_mat - centers[cluster_ids], axis=1)

        result_pdf["dist_to_centroid"] = dists.astype(np.float32)
        result_pdf["name"] = [
            decode_name.get(int(v), f"#{v}") for v in result_pdf["restaurant_name"]
        ]
        result_pdf["scored_at"] = result_pdf["kafka_ts"].astype(str)

        out = result_pdf[["row_id", "name", "cluster", "dist_to_centroid", "scored_at"]]

        n_priority = len(priority_pdf)
        n_normal   = len(normal_pdf)
        print(f"\n Batch {epoch_id}  ({len(out)} registros | "
              f"prioridad: {n_priority}, normal: {n_normal})")
        print(out.to_string(index=False, max_rows=10))

        out.to_csv(STREAM_OUT,
                   mode="a",
                   header=write_header[0],
                   index=False)
        write_header[0] = False

        seen_row_ids.update(new_priority_ids)

    query = (combined_stream.writeStream
             .outputMode("append")
             .foreachBatch(score_batch)
             .trigger(processingTime="5 seconds")
             .start())

    print(f"\nStreaming scorer arrancado (normal + prioridad).")
    print(f"  Topics  : {TOPIC}, {PRIORITY_TOPIC}  ({BOOTSTRAP})")
    print(f"  Salida  : {STREAM_OUT}")
    print(f"  Vistos  : {len(seen_row_ids)} row_ids cargados desde disco")
    print("  Ctrl+C para parar.\n")

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        query.stop()
        spark.stop()
        print("\nParado.")


if __name__ == "__main__":
    # pandas se importa aquí para que esté disponible dentro de score_batch
    import pandas as pd
    main()
