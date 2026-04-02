FROM apache/airflow:3.1.8

USER root

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

USER airflow

RUN uv pip install --no-cache \
    "tritonclient[grpc]==2.41.0" \
    "grpcio==1.67.1"

