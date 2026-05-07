#!/usr/bin/env python

from confluent_kafka import Consumer

if __name__ == '__main__':

    config = {
        # User-specific properties that you must set
        'bootstrap.servers': 'localhost:9092',
        #'bootstrap.servers': 'docker02.aulas.eif.urjc.es:9094',
        #'bootstrap.servers': '10.110.100.77',
        # Fixed properties
        'group.id':          'kafka-python-getting-started',
        'enable.auto.commit': 'false',
        'auto.offset.reset': 'earliest'
    }

    # Create Consumer instance
    consumer = Consumer(config)

    # Subscribe to topic
    topic = "purchases"
    consumer.subscribe([topic])
    print(consumer.assignment())

    # Poll for new messages from Kafka and print them.
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                # Initial message consumption may take up to
                # `session.timeout.ms` for the consumer group to
                # rebalance and start consuming
                print("Waiting...")
            elif msg.error():
                print("ERROR: %s".format(msg.error()))
            else:
                # Extract the (optional) key and value, and print.
                print("Consumed event from topic {topic}: key = {key:12} value = {value:12}".format(
                    topic=msg.topic(), key=msg.key().decode('utf-8'), value=msg.value().decode('utf-8')))
    except KeyboardInterrupt:
        consumer.unsubscribe()
    finally:
        # Leave group and commit final offsets
        consumer.close()