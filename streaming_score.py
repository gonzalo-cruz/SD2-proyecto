"""
Spark Structured Streaming — KMeans inference on Kafka topic 'restaurants'.

Pipeline:
  1. Read schema from 'restaurants_schema' topic (sent by tasks/producer.py)
  2. Stream micro-batches from 'restaurants' topic
  3. Rename dotted column names to Spark-safe sanitized names
  4. Apply VectorAssembler + trained KMeansModel
  5. Compute distance to assigned centroid (numpy, in driver)
  6. Append scored records to data/streaming/results.csv (read by app.py live feed)
  7. Print summary to console

Run producer first:
    python tasks/producer.py

Then start scorer (in a separate terminal):
    python streaming_score.py

The Streamlit app picks up new rows from data/streaming/results.csv automatically.
"""
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
    with open(FEATURE_COLS_JSON) as f:
        feature_cols = json.load(f)            # sanitized names already

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

    # ── 1. Read schema from metadata topic ───────────────────────────────────
    print(f"Reading schema from '{TOPIC}_schema' ...")
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
        print("ERROR: no schema found in 'restaurants_schema'. "
              "Run tasks/producer.py first.")
        spark.stop()
        return

    type_dict = orjson.loads(raw_schema[0][0])
    type_dict["row_id"] = "label_encoded"   # added by producer at send time
    spark_schema = map_spark_types(type_dict)
    print(f"  Schema loaded: {len(type_dict)} columns")

    # ── 2. Load ML artefacts ─────────────────────────────────────────────────
    model     = KMeansModel.load(str(KMEANS_MODEL))
    centers   = np.array(model.clusterCenters(), dtype=np.float32)  # always consistent with model
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features",
                                handleInvalid="keep")

    # ── 3. Streaming read ────────────────────────────────────────────────────
    raw_stream = (spark.readStream
                  .format("kafka")
                  .option("kafka.bootstrap.servers", BOOTSTRAP)
                  .option("subscribe", TOPIC)
                  .option("startingOffsets", "earliest")
                  .option("maxOffsetsPerTrigger", 500)
                  .load())

    # Parse JSON → columns with original names (may contain dots)
    parsed = (raw_stream
              .selectExpr("CAST(value AS STRING) as json_payload",
                          "timestamp as kafka_ts")
              .select(from_json(col("json_payload"), spark_schema).alias("d"),
                      col("kafka_ts"))
              .select("d.*", "kafka_ts"))

    # Rename dotted column names → sanitized so Spark SQL stops misinterpreting them
    rename_map = {c: sanitize_col(c) for c in type_dict if "." in c}
    if rename_map:
        parsed = parsed.select(
            [col(f"`{c}`").alias(sanitize_col(c)) if "." in c else col(c)
             for c in parsed.columns]
        )

    # ── 4. foreachBatch: score + write ───────────────────────────────────────
    STREAM_OUT.parent.mkdir(parents=True, exist_ok=True)
    write_header = [not STREAM_OUT.exists()]

    def score_batch(batch_df, epoch_id):
        if batch_df.isEmpty():
            return

        featured = assembler.transform(batch_df)
        scored   = model.transform(featured)

        pdf = scored.select(
            "row_id", "restaurant_name", "kafka_ts", "cluster", "features"
        ).toPandas()

        feat_mat    = np.array(pdf["features"].tolist(), dtype=np.float32)
        cluster_ids = pdf["cluster"].values.astype(int)
        dists       = np.linalg.norm(feat_mat - centers[cluster_ids], axis=1)

        pdf["dist_to_centroid"] = dists.astype(np.float32)
        pdf["name"] = [
            decode_name.get(int(v), f"#{v}") for v in pdf["restaurant_name"]
        ]
        pdf["scored_at"] = pdf["kafka_ts"].astype(str)

        out = pdf[["row_id", "name", "cluster", "dist_to_centroid", "scored_at"]]

        print(f"\n--- Batch {epoch_id}  ({len(out)} records) ---")
        print(out.to_string(index=False, max_rows=10))

        out.to_csv(STREAM_OUT,
                   mode="a",
                   header=write_header[0],
                   index=False)
        write_header[0] = False

    # ── 5. Start query ────────────────────────────────────────────────────────
    query = (parsed.writeStream
             .outputMode("append")
             .foreachBatch(score_batch)
             .trigger(processingTime="5 seconds")
             .start())

    print(f"\nStreaming scorer running.")
    print(f"  Topic   : {TOPIC}  ({BOOTSTRAP})")
    print(f"  Output  : {STREAM_OUT}")
    print("  Press Ctrl+C to stop.\n")

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        query.stop()
        spark.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
