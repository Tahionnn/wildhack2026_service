from datetime import datetime, timedelta
import logging
import os
import io

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
        filepath="/opt/airflow/data/test_team_track.parquet",
        poke_interval=10,
        timeout=60 * 5,
        mode="poke",
    )

    wait_for_history = FileSensor(
        task_id="wait_for_history_file",
        filepath="/opt/airflow/data/train_team_track.parquet",
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
                "history":"/opt/airflow/data/train_team_track.parquet",
                "run_id": context["run_id"],
            }

        @task
        def make_features(meta: dict):
            import pandas as pd
            from libs.feature_extraction.add_features import add_features

            run_id = meta["run_id"]
            os.makedirs(f"/opt/airflow/data/{run_id}", exist_ok=True)

            TARGET_COL = "target_2h"
            INPUT_PATH = meta["history"] 
            
            df = pd.read_parquet(INPUT_PATH)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            df_with_features = add_features(df, TARGET_COL)
            
            if TARGET_COL in df_with_features.columns:
                df_with_features = df_with_features.drop(columns=[TARGET_COL])


            last_ts = df_with_features["timestamp"].max()
            test_fe = df_with_features[df_with_features["timestamp"] == last_ts].copy()
            
            out_path = f"/opt/airflow/data/{run_id}/features_prepared.parquet"
            test_fe.to_parquet(out_path, index=False)
            
            logger.info(f"Features shape: {df_with_features.shape}")
            logger.info(f"Columns: {df_with_features.columns.tolist()[:10]}...")
            
            return {
                "original": meta["input"], 
                "featured": out_path,
                "run_id": run_id,
            }

        raw = extract_raw_data()
        return make_features(raw)

    @task_group(group_id="inference")
    def model_inference(feature_data):

        @task
        def run_inference(data: dict):
            import json
            import pandas as pd
            import numpy as np
            import tritonclient.grpc as grpcclient
            from tritonclient.utils import InferenceServerException
            from airflow.sdk import BaseHook

            run_id = data["run_id"]
            featured_path = data["featured"]

            if not os.path.exists(featured_path):
                raise FileNotFoundError(f"Features file not found: {featured_path}")

            df = pd.read_parquet(featured_path)

            with open("/opt/airflow/data/feature_cols.json") as f:
                meta = json.load(f)
            feature_cols = meta["feature_cols"]

            missing_cols = set(feature_cols) - set(df.columns)
            if missing_cols:
                raise RuntimeError(f"Missing feature columns in input: {missing_cols}")

            if df.shape[0] == 0:
                raise RuntimeError(f"No rows to predict in {featured_path}")

            X = df[feature_cols].astype("float32").to_numpy()

            conn = BaseHook.get_connection("triton_grpc_conn")
            url = f"{conn.host}:{conn.port}"
            client = grpcclient.InferenceServerClient(url)

            if not client.is_server_live():
                raise RuntimeError("Triton server is not live")
            if not client.is_model_ready("ridge_catboost_ensemble"):
                raise RuntimeError("Model ridge_catboost_ensemble is not ready")

            print(f"[ridge_catboost_ensemble] Sending INPUT_ARRAY to Triton, shape: {X.shape}")

            inputs = grpcclient.InferInput("INPUT_ARRAY", X.shape, "FP32")
            inputs.set_data_from_numpy(X)

            outputs = [grpcclient.InferRequestedOutput("OUTPUT_PREDS")]

            try:
                response = client.infer(
                    model_name="ridge_catboost_ensemble",
                    inputs=[inputs],
                    outputs=outputs,
                )
            except InferenceServerException as e:
                raise RuntimeError(f"Triton inference failed: {e.message()}")
            
            if "OUTPUT_PREDS" not in response.get_response().outputs:
                raise RuntimeError("No OUTPUT_PREDS returned by Triton!")

            y_pred = response.as_numpy("OUTPUT_PREDS")
            if y_pred is None or y_pred.size == 0:
                raise RuntimeError("Received empty prediction array from Triton!")

            if y_pred is None or y_pred.size == 0:
                raise RuntimeError("[ridge_catboost_ensemble] Triton returned empty predictions!")

            result = df[["route_id", "timestamp"]].copy()
            result["target_2h"] = y_pred[:, 3]

            out_path = f"/opt/airflow/data/{run_id}/predictions.csv"
            result.to_csv(out_path, index=False)
            print(f"[ridge_catboost_ensemble] Inference done: {len(result)} rows, saved to {out_path}")

            return {
                "predictions": out_path,
                "run_id": run_id,
                "original": data["original"],
            }

        @task
        def post_proccess_preds(meta: dict):
            import pandas as pd

            input_df = pd.read_parquet(meta["original"])
            preds = pd.read_csv(meta["predictions"])

            submission = input_df.copy()
            submission = submission.merge(preds, on=["route_id", "timestamp"], how="left")
            submission['y_pred'] = submission['target_2h']
            
            out_path = f"/opt/airflow/data/{meta['run_id']}/submission.csv"
            submission.to_csv(out_path, index=False)
            
            logger.info(f"Post-processing done, saved to {out_path}")
            
            return {
                "predictions": meta["predictions"],
                "submission": out_path,
                "run_id": meta["run_id"],
                "original": meta["original"],
            }
        
        run = run_inference(feature_data)
        return post_proccess_preds(run)

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

    [wait_for_input, wait_for_history] >> features >> inference >> result >> pub


transport_pipeline()
