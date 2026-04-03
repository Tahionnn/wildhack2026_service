#!/usr/bin/env python3
"""
Тестовый gRPC клиент для загрузки CSV-файла на сервер.
Использует клиентский стриминг.

Пример запуска:
    python test_client.py --file data.csv --server localhost:50051 --chunk-size 65536
"""

import argparse
import sys
import grpc

# Импортируем сгенерированные protobuf-файлы
import api.api_pb2 as csv_upload_pb2
import api.api_pb2_grpc as csv_upload_pb2_grpc

def upload_csv(stub, file_path, chunk_size):
    """
    Генератор, который читает файл и выдаёт чанки данных.
    """
    def generate_chunks():
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield csv_upload_pb2.CSVChunk(content=chunk)

    try:
        # Отправляем поток чанков и получаем ответ от сервера
        response = stub.UploadCSV(generate_chunks())
        print(f"[✓] Статус: {response.message}")
        print(f"[✓] Обработано строк: {response.rows_processed}")
        return True
    except grpc.RpcError as e:
        print(f"[✗] Ошибка gRPC: {e.code()} - {e.details()}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[✗] Неожиданная ошибка: {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="result.csv", help="Путь к CSV-файлу (по умолчанию result.csv)")
    parser.add_argument("--server", default="localhost:50051")
    parser.add_argument("--chunk-size", type=int, default=64 * 1024)
    args = parser.parse_args()

    # Создаём канал и заглушку
    channel = grpc.insecure_channel(args.server)
    stub = csv_upload_pb2_grpc.CSVUploadServiceStub(channel)

    print(f"Загружаем файл '{args.file}' на сервер {args.server}...")
    print(f"Размер чанка: {args.chunk_size} байт")
    success = upload_csv(stub, args.file, args.chunk_size)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()