import json
import os
from time import sleep

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast, desc, get_json_object

os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-21-openjdk-amd64"

KAFKA_BROKER    = "localhost:9092"
KAFKA_TOPIC     = "restaurants"
ENCODINGS_PATH  = "data/processed/encodings.json"
MAX_OFFSETS_APPEND   = 10_000
MAX_OFFSETS_COMPLETE = 200_000

conf = SparkConf()
conf.set("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1")


def make_stream(spark, max_offsets):
    return (spark.readStream
            .format("kafka")
            .option("kafka.bootstrap.servers", KAFKA_BROKER)
            .option("subscribe", KAFKA_TOPIC)
            .option("startingOffsets", "earliest")
            .option("maxOffsetsPerTrigger", max_offsets)
            .load()
            .selectExpr("CAST(value AS STRING)"))


def save_batch(path):
    def _write(batch_df, _epoch_id):
        batch_df.toPandas().to_csv(path, index=False)
        print(f"=== {path} ===")
        batch_df.show()
    return _write


if __name__ == "__main__":
    spark = (SparkSession.builder
             .appName("KafkaQueries")
             .master("local[2]")
             .config(conf=conf)
             .getOrCreate())

    with open(ENCODINGS_PATH) as f:
        enc = json.load(f)

    name_lookup = spark.createDataFrame(
        [(name, code) for name, code in enc["restaurant_name"].items()],
        ["restaurant_name", "name_code"],
    )
    country_lookup = spark.createDataFrame(
        [(name, code) for name, code in enc["country"].items()],
        ["country_name", "country_code"],
    )
    city_lookup = spark.createDataFrame(
        [(name, code) for name, code in enc["city"].items()],
        ["city_name", "city_code"],
    )

    # Consulta 1 — outputMode "append"                                    #
    # Extrae campos clave del JSON de cada registro según llega,          #
    # decodificando los valores label-encoded con encodings.json.         #
    s1 = make_stream(spark, MAX_OFFSETS_APPEND)
    parsed = (s1
              .select(
                  get_json_object("value", "$.restaurant_name").cast("int").alias("name_code"),
                  get_json_object("value", "$.country").cast("int").alias("country_code"),
                  get_json_object("value", "$.city").cast("int").alias("city_code"),
                  get_json_object("value", "$.cuisines").alias("cuisines"),
              )
              .join(broadcast(name_lookup), "name_code", "left")
              .join(broadcast(country_lookup), "country_code", "left")
              .join(broadcast(city_lookup), "city_code", "left")
              .select("restaurant_name", "country_name", "city_name", "cuisines"))
    q1 = (parsed.writeStream
          .foreachBatch(save_batch("salida1.txt"))
          .outputMode("append")
          .trigger(processingTime="10 seconds")
          .start())
    sleep(12)
    q1.stop()

    # Consulta 2 — outputMode "complete"                                  
    # Conteo de restaurantes por país sobre el dataset completo.          
    # MAX_OFFSETS_COMPLETE + sleep=70s cubre los ~1M registros del topic. 
    s2 = make_stream(spark, MAX_OFFSETS_COMPLETE)
    by_country = (s2
                  .select(get_json_object("value", "$.country").cast("int").alias("country_code"))
                  .join(country_lookup, "country_code", "left")
                  .groupBy("country_name")
                  .count()
                  .orderBy(desc("count")))
    q2 = (by_country.writeStream
          .foreachBatch(save_batch("salida2.txt"))
          .outputMode("complete")
          .trigger(processingTime="10 seconds")
          .start())
    sleep(70)
    q2.stop()

    # Consulta 3 — outputMode "complete"                                  
    # Ranking de cocinas más frecuentes en el dataset completo.           
    s3 = make_stream(spark, MAX_OFFSETS_COMPLETE)
    by_cuisine = (s3
                  .select(get_json_object("value", "$.cuisines").alias("cuisines"))
                  .groupBy("cuisines")
                  .count()
                  .orderBy(desc("count")))
    q3 = (by_cuisine.writeStream
          .foreachBatch(save_batch("salida3.txt"))
          .outputMode("complete")
          .trigger(processingTime="10 seconds")
          .start())
    sleep(70)
    q3.stop()

    # Consulta 4 — outputMode "append"                                    
    # Restaurantes con valoración 3.5 estrellas (máximo presente tras     
    # el preprocesado). avg_rating__3.5 vale 1 si el restaurante tiene    
    # esa puntuación (OHE).                                               
    s4 = make_stream(spark, MAX_OFFSETS_APPEND)
    top_rated = (s4
                 .select(
                     get_json_object("value", "$.restaurant_name").cast("int").alias("name_code"),
                     get_json_object("value", "$.cuisines").alias("cuisines"),
                     get_json_object("value", "$['avg_rating__3.5']").cast("int").alias("rating_35"),
                 )
                 .filter("rating_35 = 1")
                 .join(broadcast(name_lookup), "name_code", "left")
                 .select("restaurant_name", "cuisines", "rating_35"))
    q4 = (top_rated.writeStream
          .foreachBatch(save_batch("salida4.txt"))
          .outputMode("append")
          .trigger(processingTime="10 seconds")
          .start())
    sleep(12)
    q4.stop()

    # Consulta 5 — outputMode "complete"                                  
    # Top 20 ciudades con más restaurantes en el dataset completo.        
    s5 = make_stream(spark, MAX_OFFSETS_COMPLETE)
    by_city = (s5
               .select(get_json_object("value", "$.city").cast("int").alias("city_code"))
               .join(city_lookup, "city_code", "left")
               .groupBy("city_name")
               .count()
               .orderBy(desc("count"))
               .limit(20))
    q5 = (by_city.writeStream
          .foreachBatch(save_batch("salida5.txt"))
          .outputMode("complete")
          .trigger(processingTime="10 seconds")
          .start())
    sleep(70)
    q5.stop()

    # Consulta 6 — outputMode "append"                                    
    # Restaurantes con las tres opciones dietéticas disponibles:          
    # vegetariano, vegano y sin gluten (columnas OHE = 1).                
    s6 = make_stream(spark, MAX_OFFSETS_APPEND)
    diet_friendly = (s6
                     .select(
                         get_json_object("value", "$.restaurant_name").cast("int").alias("name_code"),
                         get_json_object("value", "$.country").cast("int").alias("country_code"),
                         get_json_object("value", "$.cuisines").alias("cuisines"),
                         get_json_object("value", "$['vegetarian_friendly__Y']").cast("int").alias("vegetarian"),
                         get_json_object("value", "$['vegan_options__Y']").cast("int").alias("vegan"),
                         get_json_object("value", "$['gluten_free__Y']").cast("int").alias("gluten_free"),
                     )
                     .filter("vegetarian = 1 AND vegan = 1 AND gluten_free = 1")
                     .join(broadcast(name_lookup), "name_code", "left")
                     .join(broadcast(country_lookup), "country_code", "left")
                     .select("restaurant_name", "country_name", "cuisines"))
    q6 = (diet_friendly.writeStream
          .foreachBatch(save_batch("salida6.txt"))
          .outputMode("append")
          .trigger(processingTime="10 seconds")
          .start())
    sleep(12)
    q6.stop()

    spark.stop()
