from faststream import FastStream
from faststream.kafka import KafkaBroker
from config import BROKER_URL

# --- Broker Setup ---
broker = KafkaBroker(
    BROKER_URL,
    enable_idempotence=True
)
app = FastStream(broker)

# --- Publishers for the three topics ---
# The `topic` parameter defines the destination topic in the broker.
tms_publisher = broker.publisher(topic="TMS")
wms_publisher = broker.publisher(topic="WMS")
yms_publisher = broker.publisher(topic="YMS")