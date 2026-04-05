def upload_csv(file_path: str, target: str):
    import grpc
    from airflow.sdk import BaseHook
    from . import api_pb2, api_pb2_grpc

    CHUNK_SIZE = 1024 * 1024

    def generate_chunks():
        with open(file_path, "rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                yield api_pb2.CSVChunk(content=chunk)

    with grpc.insecure_channel(target) as channel:
        stub = api_pb2_grpc.CSVUploadServiceStub(channel)

        response = stub.UploadCSV(generate_chunks())