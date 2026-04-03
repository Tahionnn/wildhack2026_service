import os
from dotenv import load_dotenv, find_dotenv

env_path = find_dotenv()
if env_path:
    print(f"Найден .env файл: {env_path}")
else:
    print("Файл .env не найден")

load_dotenv(env_path)

GRPC_PORT = os.getenv("GRPC_PORT", 50051)
BROKER_URL = os.getenv("BROKER_URL", "localhost:9092")

#TODO добавить переменные в .env