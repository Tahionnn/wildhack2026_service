# broker_setup.py
import os
from faststream import FastStream
from faststream.kafka import KafkaBroker  # or RabbitBroker

# --- Broker Setup ---
broker = KafkaBroker(
    os.getenv("BROKER_URL", "localhost:9092")  # Use environment variables for configuration
)
app = FastStream(broker)

# --- Publishers for the three topics ---
# The `topic` parameter defines the destination topic in the broker.
tms_publisher = broker.publisher(topic="TMS")
wms_publisher = broker.publisher(topic="WMS")
yms_publisher = broker.publisher(topic="YMS")