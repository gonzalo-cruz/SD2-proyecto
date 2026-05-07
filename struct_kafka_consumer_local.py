#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
from time import sleep

from pyspark import SparkConf
from pyspark.sql import SparkSession

os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-21-openjdk-amd64"

KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC  = "restaurants"
MAX_OFFSETS  = 5000

conf = SparkConf()
conf.set("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1")

if __name__ == "__main__":
    spark = (SparkSession.builder
             .appName("KafkaStructuredStreaming")
             .master("local[2]")
             .config(conf=conf)
             .getOrCreate())

    input_data = (spark.readStream
                  .format("kafka")
                  .option("kafka.bootstrap.servers", KAFKA_BROKER)
                  .option("subscribe", KAFKA_TOPIC)
                  .option("startingOffsets", "earliest")
                  .option("maxOffsetsPerTrigger", MAX_OFFSETS)
                  .load()
                  .selectExpr("CAST(value AS STRING)"))

    describe_query = (input_data.writeStream
                      .queryName("raw_data")
                      .format("memory")
                      .outputMode("append")
                      .trigger(processingTime="10 seconds")
                      .start())

    for x in range(5):
        spark.sql("SELECT * FROM raw_data").show()
        sleep(3)

    describe_query.stop()
    spark.stop()
