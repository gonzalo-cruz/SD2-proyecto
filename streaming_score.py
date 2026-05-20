# Consumidor de Spark Structured Streaming que clasifica los restaurantes
# que llegan por Kafka en tiempo real usando el modelo KMeans entrenado.
#
#   1. Lee el schema del topic 'restaurants_schema' (enviado por producer.py)
#   2. Consume el topic 'restaurants' en micro-batches de 5 segundos
#   3. Por cada batch: ensambla el vector de features, aplica KMeans y
#      calcula la distancia al centroide del cluster asignado
#   4. Escribe los resultados en data/streaming/results.csv
#      (la app los lee cada 5s para actualizar el tab "En vivo")
import json
import os
import numpy as np
import orjson
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    StructType, StructField,
    FloatType, DoubleType, IntegerType, LongType, StringType,
)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeansModel

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"

BASE_DIR          = Path(__file__).parent
ENCODINGS_JSON    = BASE_DIR / "data" / "processed" / "encodings.json"
FEATURE_COLS_JSON = BASE_DIR / "models" / "feature_cols.json"
KMEANS_MODEL      = BASE_DIR / "models" / "kmeans_spark"
STREAM_OUT        = BASE_DIR / "data" / "streaming" / "results.csv"

TOPIC     = "restaurants"
BOOTSTRAP = "localhost:9092"


def sanitize_col(name: str) -> str:
    # Los nombres de columna con '.' los renombramos igual que en train_model.py
    # para que el VectorAssembler encuentre las mismas columnas
    return name.replace(".", "_")


def map_spark_types(type_dict: dict) -> StructType:
    # Convertimos el diccionario de tipos del pipeline al schema de Spark
    # necesario para parsear los mensajes JSON de Kafka
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
    # Cargamos los nombres de features que uso el entrenamiento para garantizar
    # que el VectorAssembler aplique exactamente las mismas columnas
    with open(FEATURE_COLS_JSON) as f:
        feature_cols = json.load(f)

    # Cargamos el encoding de nombres para mostrar el nombre real del restaurante
    with open(ENCODINGS_JSON) as f:
        enc = json.load(f)
    decode_name = {int(v): k for k, v in enc["restaurant_name"].items()}

    spark = (SparkSession.builder
             .appName("RestaurantStreamScorer")
             .master("local[*]")
             .config("spark.jars.packages",
                     "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1")
             .config("spark.driver.memory", "2g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # Leer el schema del topic de metadatos 
    # producer.py envia el schema antes de los datos para que el consumidor
    # sepa como parsear los mensajes JSON
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
        print("ERROR: no se encontro schema en 'restaurants_schema'. "
              "Ejecuta tasks/producer.py primero.")
        spark.stop()
        return

    type_dict = orjson.loads(raw_schema[0][0])
    type_dict["row_id"] = "label_encoded"   # el producer añade row_id a cada mensaje
    spark_schema = map_spark_types(type_dict)
    print(f"  Schema cargado: {len(type_dict)} columnas")

    # Cargar el modelo KMeans entrenado 
    model   = KMeansModel.load(str(KMEANS_MODEL))
    # Leemos los centroides directamente del modelo para asegurar consistencia
    centers = np.array(model.clusterCenters(), dtype=np.float32)
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features",
                                handleInvalid="keep")

    # Iniciar la lectura del stream de Kafka 
    raw_stream = (spark.readStream
                  .format("kafka")
                  .option("kafka.bootstrap.servers", BOOTSTRAP)
                  .option("subscribe", TOPIC)
                  .option("startingOffsets", "earliest")
                  .option("maxOffsetsPerTrigger", 500)
                  .load())

    # Parseamos el JSON de cada mensaje usando el schema leido antes
    parsed = (raw_stream
              .selectExpr("CAST(value AS STRING) as json_payload",
                          "timestamp as kafka_ts")
              .select(from_json(col("json_payload"), spark_schema).alias("d"),
                      col("kafka_ts"))
              .select("d.*", "kafka_ts"))

    # Renombramos columnas con '.' porque Spark las interpreta como acceso a struct field
    rename_map = {c: sanitize_col(c) for c in type_dict if "." in c}
    if rename_map:
        parsed = parsed.select(
            [col(f"`{c}`").alias(sanitize_col(c)) if "." in c else col(c)
             for c in parsed.columns]
        )

    # Funcion que se ejecuta en cada micro-batch 
    STREAM_OUT.parent.mkdir(parents=True, exist_ok=True)
    write_header = [not STREAM_OUT.exists()]  # escribimos cabecera solo la primera vez

    def score_batch(batch_df, epoch_id):
        if batch_df.isEmpty():
            return

        # Aplicamos el pipeline de ML: ensamblamos features y predecimos cluster
        featured = assembler.transform(batch_df)
        scored   = model.transform(featured)

        pdf = scored.select(
            "row_id", "restaurant_name", "kafka_ts", "cluster", "features"
        ).toPandas()

        # Calculamos la distancia euclidea al centroide del cluster asignado
        feat_mat    = np.array(pdf["features"].tolist(), dtype=np.float32)
        cluster_ids = pdf["cluster"].values.astype(int)
        dists       = np.linalg.norm(feat_mat - centers[cluster_ids], axis=1)

        pdf["dist_to_centroid"] = dists.astype(np.float32)
        pdf["name"] = [
            decode_name.get(int(v), f"#{v}") for v in pdf["restaurant_name"]
        ]
        pdf["scored_at"] = pdf["kafka_ts"].astype(str)

        out = pdf[["row_id", "name", "cluster", "dist_to_centroid", "scored_at"]]

        print(f"\n Batch {epoch_id}  ({len(out)} registros)")
        print(out.to_string(index=False, max_rows=10))

        # Anadimos al CSV (mode="a") para que la app vea el historico completo
        out.to_csv(STREAM_OUT,
                   mode="a",
                   header=write_header[0],
                   index=False)
        write_header[0] = False

    # Arrancar el stream 
    query = (parsed.writeStream
             .outputMode("append")
             .foreachBatch(score_batch)
             .trigger(processingTime="5 seconds")
             .start())

    print(f"\nStreaming scorer arrancado.")
    print(f"  Topic   : {TOPIC}  ({BOOTSTRAP})")
    print(f"  Salida  : {STREAM_OUT}")
    print("  Ctrl+C para parar.\n")

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        query.stop()
        spark.stop()
        print("\nParado.")


if __name__ == "__main__":
    main()
