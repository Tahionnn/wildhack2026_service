from datetime import datetime, timedelta
import logging
import os

from airflow.sdk import dag, task, task_group
from airflow.sdk import BaseHook
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.standard.sensors.filesystem import FileSensor
from docker.types import Mount


logger = logging.getLogger(__name__)

default_args = {
    "owner": "mlops_team",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


@dag(
    dag_id="transport_calling_system",
    default_args=default_args,
    schedule="0 */4 * * *",
    max_active_runs=1,
    catchup=False,
    tags=["transport"],
)
def transport_pipeline():
    wait_for_input = FileSensor(
        task_id="wait_for_input_file",
        filepath=f"/opt/airflow/data/test_team_track.parquet",
        poke_interval=10,
        timeout=60 * 5,
        mode="poke",
    )

    @task_group(group_id="features")
    def feature_engineering():

        @task
        def extract_raw_data(**context):
            return {
                "input": "/opt/airflow/data/test_team_track.parquet",
                "run_id": context["run_id"]
            }

        @task
        def make_features(meta: dict):
            import pandas as pd

            run_id = meta["run_id"]

            os.makedirs(f"/opt/airflow/data/{run_id}", exist_ok=True)

            df = pd.read_parquet(meta["input"])

            output_path = f"/opt/airflow/data/{run_id}/features_prepared.parquet"
            df.to_parquet(output_path)

            return {
                "original": meta["input"],
                "featured": output_path,
                "run_id": run_id
            }

        raw = extract_raw_data()
        return make_features(raw)

    @task_group(group_id="inference")
    def model_inference(feature_data):

        @task
        def run_inference(data: dict):
            import pandas as pd
            import numpy as np
            import tritonclient.grpc as grpcclient
            from tritonclient.utils import InferenceServerException

            run_id = data["run_id"]

            featured_path = data["featured"]

            if not os.path.exists(featured_path):
                raise FileNotFoundError(featured_path)

            df = pd.read_parquet(featured_path)

            exclude_cols = ["route_id", "timestamp", "target_2h"]
            feature_cols = [c for c in df.columns if c not in exclude_cols]

            input_tensor = df[feature_cols].values.astype("float32")

            conn = BaseHook.get_connection("triton_grpc_conn")
            url = f"{conn.host}:{conn.port}"

            client = grpcclient.InferenceServerClient(url)

            if not client.is_server_live():
                raise RuntimeError("Triton not live")

            inputs = [grpcclient.InferInput("INPUT0", input_tensor.shape, "FP32")]
            inputs[0].set_data_from_numpy(input_tensor)

            outputs = [grpcclient.InferRequestedOutput("OUTPUT0")]

            try:
                response = client.infer(
                    model_name="identity_model",
                    inputs=inputs,
                    outputs=outputs,
                )
            except InferenceServerException as e:
                raise RuntimeError(e.message())

            y_pred = response.as_numpy("OUTPUT0").flatten()

            result = df[["route_id", "timestamp"]].copy()
            result["target_2h"] = y_pred

            out_path = f"/opt/airflow/data/{run_id}/predictions.csv"
            result.to_csv(out_path, index=False)

            return {
                "predictions": out_path,
                "run_id": run_id,
                "original": data["original"],
            }

        return run_inference(feature_data)
    
    @task_group(group_id="optimization")
    def optimization(feature_data, inference_data):

        @task
        def merge_inputs(features: dict, inference: dict):
            return {
                "original": features["original"],
                "run_id": features["run_id"],
                "predictions": inference["predictions"],
            }

        merged = merge_inputs(feature_data, inference_data)

        @task
        def prepare_data(data: dict):
            import pandas as pd

            run_id = data["run_id"]

            raw = pd.read_parquet(data["original"])
            pred = pd.read_csv(data["predictions"])
            route_map = pd.read_csv("/opt/airflow/data/route_map.csv")

            raw["timestamp"] = pd.to_datetime(raw["timestamp"])
            pred["timestamp"] = pd.to_datetime(pred["timestamp"])

            df = raw.merge(pred, on=["route_id", "timestamp"], how="left")
            df = df.merge(route_map, on="route_id", how="left")

            df = df.rename(columns={"target_2h": "predicted_demand"})

            out = f"/opt/airflow/data/{run_id}/solver_input.csv"
            df.to_csv(out, index=False)

            return {
                "solver_input": out,
                "run_id": run_id,
            }

        prep = prepare_data(merged)

        @task(task_id="build_solver_paths")
        def build_solver_paths(data: dict):
            run_id = data["run_id"]

            return {
                "input": f"{run_id}/solver_input.csv",
                "output": f"{run_id}/result.csv",
                "config": "config.json",
                "run_id": run_id,
            }

        paths = build_solver_paths(prep)

        solve = DockerOperator(
            task_id="solve_transport",
            image="transport-optimizer:latest",
            command="""
            python /app/main.py 
            /data/{{ ti.xcom_pull(task_ids='optimization.build_solver_paths')['config'] }} 
            /data/{{ ti.xcom_pull(task_ids='optimization.build_solver_paths')['input'] }} 
            /data/{{ ti.xcom_pull(task_ids='optimization.build_solver_paths')['output'] }}
            """,
            mounts=[
                Mount(
                    source=os.environ["DATA_PATH"],
                    target="/data",
                    type="bind",
                )
            ],
            docker_url="unix://var/run/docker.sock",
            auto_remove="success",
            mount_tmp_dir=False,
            do_xcom_push=False,
            user="root",  # Need to fix this shit before deadline...
            retries=0,
        )

        @task
        def get_result_path(data: dict):
            run_id = data["run_id"]
            return {
                "result_path": f"/opt/airflow/data/{run_id}/result.csv",
                "run_id": run_id,
            }

        result = get_result_path(paths)

        prep >> paths >> solve >> result
        return result

    @task
    def publish(data: dict):
        from libs.grpc.client import upload_csv

        run_id = data["run_id"]

        conn = BaseHook.get_connection("event_producer_grpc_conn")
        target = f"{conn.host}:{conn.port}"
        result_path = f"/opt/airflow/data/{run_id}/result.csv"
        
        return upload_csv(result_path, target)

    features = feature_engineering()
    inference = model_inference(features)
    result = optimization(features, inference)
    pub = publish(result)

    wait_for_input >> features >> inference >> result >> pub


transport_pipeline()
