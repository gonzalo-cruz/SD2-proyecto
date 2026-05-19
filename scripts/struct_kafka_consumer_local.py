import os
import orjson
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, desc as desc_offset
from pyspark.sql.types import *

# Configuración de entorno
os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-21-openjdk-amd64"

def map_spark_types(type_dict):
    """Mapea los tipos de type_dict_encoding.json a tipos de Spark SQL."""
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

if __name__ == "__main__":
    # Spark 4.1.1 compatible con el conector 0-10_2.12
    spark = (SparkSession.builder
        .appName("KafkaUnboundedTable")
        .master("local[*]")
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1")
        .getOrCreate())

    bootstrap_servers = "localhost:9092"
    topic_name = "restaurants"

    # 1. Obtener el esquema desde Kafka (Lectura Batch)
    log_schema = (spark.read
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", f"{topic_name}_schema")
        .option("startingOffsets", "earliest")
        .load())

    # Recuperar el último mensaje de esquema enviado
    raw_schema_json = (log_schema
        .selectExpr("CAST(value AS STRING)", "offset")
        .orderBy(desc_offset("offset"))
        .limit(1)
        .collect()[0][0])
    schema_dict = orjson.loads(raw_schema_json)
    spark_schema = map_spark_types(schema_dict)

    # 2. Iniciar Stream de datos
    df_stream = (spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", topic_name)
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", 5000)
        .load())

    # 3. Transformar JSON a columnas (Tabla Unbounded)
    unbounded_table = (df_stream
        .selectExpr("CAST(value AS STRING) as json_payload")
        .select(from_json(col("json_payload"), spark_schema).alias("data"))
        .select("data.*"))

    # 4. Guardar en fichero de texto
    output_path = "salida_stream.txt"
    first_batch = [True]

    def save_batch(batch_df, _epoch_id):
        pdf = batch_df.toPandas()
        pdf.to_csv(output_path, mode="w" if first_batch[0] else "a",
                   header=first_batch[0], index=False)
        first_batch[0] = False
        print(f"Batch escrito en {output_path} ({len(pdf)} filas)")

    query = (unbounded_table.writeStream
        .outputMode("append")
        .foreachBatch(save_batch)
        .trigger(processingTime="10 seconds")
        .start())

    query.awaitTermination(timeout=30)
    query.stop()
    spark.stop()