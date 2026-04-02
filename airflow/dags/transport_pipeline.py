from datetime import datetime, timedelta
import logging
import os

from airflow.sdk import dag, task, task_group
from airflow.sdk import BaseHook
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.docker.operators.docker import DockerOperator
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
    @task_group(group_id="features")
    def feature_engineering():

        @task
        def extract_raw_data():
            path = "/opt/airflow/data/test_team_track.parquet"
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            return path

        @task
        def make_features(input_path: str):
            import pandas as pd

            df = pd.read_parquet(input_path)

            output_path = "/opt/airflow/data/features_prepared.parquet"
            df.to_parquet(output_path)

            return {
                "original": input_path,
                "featured": output_path,
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

            out_path = "/opt/airflow/data/predictions.csv"
            result.to_csv(out_path, index=False)

            return out_path

        return run_inference(feature_data)

    @task_group(group_id="optimization")
    def optimization(feature_data, inference_path):

        @task
        def prepare_data(features: dict, pred_path: str):
            import pandas as pd

            raw = pd.read_parquet(features["original"])
            pred = pd.read_csv(pred_path)
            route_map = pd.read_csv("/opt/airflow/data/route_map.csv")

            raw['timestamp'] = pd.to_datetime(raw['timestamp'])
            pred['timestamp'] = pd.to_datetime(pred['timestamp'])

            df = raw.merge(pred, on=["route_id", "timestamp"], how="left")
            df = df.merge(route_map, on="route_id", how="left")

            df = df.rename(columns={"target_2h": "predicted_demand"})

            out = "/opt/airflow/data/solver_input.csv"
            df.to_csv(out, index=False)

            return out

        prep = prepare_data(feature_data, inference_path)

        solve = DockerOperator(
            task_id="solve_transport",
            image="transport-optimizer:latest",
            command="python /app/main.py /data/config.json /data/solver_input.csv /data/result.csv",
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
            user="root", # Need to fix this shit before deadline...
            retries=0
        )

        @task
        def publish():
            import pandas as pd

            path = "/opt/airflow/data/result.csv"
            df = pd.read_csv(path)

            return {"rows": len(df)}

        pub = publish()

        prep >> solve >> pub
        return pub

    features = feature_engineering()
    inference = model_inference(features)
    opt = optimization(features, inference)

    features >> inference >> opt

    end = EmptyOperator(task_id="end")
    opt >> end


transport_pipeline()
