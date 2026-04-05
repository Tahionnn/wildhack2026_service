import asyncio
import logging

import grpc.aio
from faststream.asgi import AsgiFastStream, make_ping_asgi

from broker_setup import broker
from api import api_pb2_grpc
from config import GRPC_PORT
from grpc_server import CSVUploadServicer

logger = logging.getLogger(__name__)

app = AsgiFastStream(
    broker,
    asyncapi_path="/docs",
    asgi_routes=[
        ("/health", make_ping_asgi(broker, timeout=5.0)),
    ],
)

grpc_server = None


@app.after_startup
async def start_grpc_server():
    global grpc_server
    grpc_server = grpc.aio.server()
    api_pb2_grpc.add_CSVUploadServiceServicer_to_server(
        CSVUploadServicer(), grpc_server
    )
    grpc_server.add_insecure_port(f"[::]:{GRPC_PORT}")
    await grpc_server.start()
    logger.info(f"gRPC server started on port {GRPC_PORT}")


@app.after_shutdown
async def stop_grpc_server():
    if grpc_server:
        await grpc_server.stop(5)
    logger.info("gRPC server stopped")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run()