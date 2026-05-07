#!/usr/bin/env python

from random import choice
from confluent_kafka import Producer

if __name__ == '__main__':

    config = {
        # User-specific properties that you must set
        #'bootstrap.servers':'docker02.aulas.eif.urjc.es:9094',
        'bootstrap.servers':'localhost:9092',
        # Fixed properties
        'acks': 'all'
    }

    try:
        # Create Producer instance
        producer = Producer(config)
    except KeyboardInterrupt:
        pass
    finally:
        pass

    
    # Optional per-message delivery callback (triggered by poll() or flush())
    # when a message has been successfully delivered or permanently
    # failed delivery (after retries).
    def delivery_callback(err, msg):
        if err:
            print('ERROR: Message failed delivery: {}'.format(err))
        else:
            print("Produced event to topic {topic}: key = {key:12} value = {value:12}".format(
                topic=msg.topic(), key=msg.key().decode('utf-8'), value=msg.value().decode('utf-8')))

    # Produce data by selecting random values from these lists.
    topic = "purchases"
    user_ids = ['eabara-2', 'jsmith-2', 'sgarcia-2', 'jbernard-2', 'htanaka-2', 'awalther-2']
    products = ['book', 'alarm clock', 't-shirts', 'gift card', 'batteries']

    count = 0
    
    try:
        for _ in range(10):
            user_id = choice(user_ids)
            product = choice(products)
            producer.produce(topic, product, user_id, callback=delivery_callback)
            count += 1
    except KeyboardInterrupt:
        pass
    finally:
        # Block until the messages are sent.
        producer.poll(10000)
        producer.flush()
    