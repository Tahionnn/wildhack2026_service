from datetime import datetime, timedelta
import logging
import os
from airflow.decorators import dag, task, task_group
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
import pandas as pd
import numpy as np
import traceback

logger = logging.getLogger(__name__)

default_args = {
    'owner': "mlops_team",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


@dag(
    dag_id="transport_calling_system",
    description="Automate transport calling system",
    default_args=default_args,
    schedule="0 */4 * * *",
    max_active_runs=1,
    tags=["transport"],
    catchup=False
)
def transport_pipeline():

    @task_group(group_id="features")
    def feature_engineering():
        
        @task
        def extract_raw_data():
            path = "/opt/airflow/data/test_team_track.parquet"
            logger.info(f"Extracting raw data from {path}")
            
            if not os.path.exists(path):
                raise FileNotFoundError(f"Raw data file not found: {path}")
            return path
        
        @task
        def make_features(input_path):
            try:
                logger.info(f"Reading parquet file from {input_path}")
                df = pd.read_parquet(input_path)
                logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
                
                output_path = "/opt/airflow/data/features_prepared.parquet"
                df.to_parquet(output_path)
                logger.info(f"Saved features to {output_path}")
                
                return {
                    "original": input_path,
                    "featured": output_path
                }
            except Exception as e:
                logger.error(f"Error in make_features: {str(e)}")
                logger.error(traceback.format_exc())
                raise

        raw_path = extract_raw_data()
        feature_df_data = make_features(raw_path)
        return feature_df_data

    @task_group(group_id="inference")
    def model_inference(feature_df_data):
        @task
        def run_inference(features_data):
            try:
                featured_path = features_data["featured"]
                logger.info(f"Running inference on {featured_path}")
                
                if not os.path.exists(featured_path):
                    raise FileNotFoundError(f"Features file not found: {featured_path}")
                
                df = pd.read_parquet(featured_path)
                logger.info(f"Loaded {len(df)} rows for inference")

                if "id" not in df.columns:
                    raise ValueError("id column is required for inference")
                
                predictions = pd.DataFrame({
                    "id": df["id"],
                    "y_pred": np.arange(len(df)) % 100
                })
                
                output_path = "/opt/airflow/data/predictions.csv"
                predictions.to_csv(output_path, index=False)
                logger.info(f"Saved {len(predictions)} predictions to {output_path}")
                
                return output_path
            except Exception as e:
                logger.error(f"Error in run_inference: {str(e)}")
                logger.error(traceback.format_exc())
                raise

        inference_result = run_inference(feature_df_data)
        return inference_result

    @task_group(group_id="optimization")
    def optimization(feature_df_data, inference_results):
        
        @task
        def prepare_data(features_data, inference_data):
            try:
                logger.info("Starting prepare_data task")
                logger.info(f"Features data: {features_data}")
                logger.info(f"Inference data: {inference_data}")
                
                
                route_map_path = "/opt/airflow/data/route_map.csv"
                if not os.path.exists(route_map_path):
                    raise FileNotFoundError(f"Route map file not found: {route_map_path}")
                
                route_map = pd.read_csv(route_map_path)
                logger.info(f"Loaded route_map with {len(route_map)} rows")
                
                if inference_data is None:
                    raise ValueError("Inference results are required for optimization")
                
                if not os.path.exists(inference_data):
                    raise FileNotFoundError(f"Inference file not found: {inference_data}")
                
                pred_df = pd.read_csv(inference_data)
                logger.info(f"Loaded predictions with {len(pred_df)} rows")
                
                if not {"id", "y_pred"}.issubset(pred_df.columns):
                    raise ValueError(f"predictions must contain columns: id, y_pred. Found: {list(pred_df.columns)}")
                
                original_path = features_data.get("original")
                if not original_path or not os.path.exists(original_path):
                    raise FileNotFoundError(f"Original data file not found: {original_path}")
                
                raw_data = pd.read_parquet(original_path)
                logger.info(f"Loaded raw data with {len(raw_data)} rows, columns: {list(raw_data.columns)}")
                
                required_cols = {"id", "route_id", "timestamp"}
                if not required_cols.issubset(raw_data.columns):
                    raise ValueError(f"raw data must contain columns: {required_cols}. Found: {list(raw_data.columns)}")
                
                
                df = raw_data.merge(pred_df, on="id", how="left")
                logger.info(f"After merge: {len(df)} rows")
                
                df = df.merge(route_map, on="route_id", how="left")
                logger.info(f"After route merge: {len(df)} rows")
                
                
                if df["y_pred"].isna().any():
                    missing_count = df["y_pred"].isna().sum()
                    raise ValueError(f"Missing predictions after merge for {missing_count} rows")
                
                
                if df["office_from_id"].isna().any():
                    missing_routes = df[df["office_from_id"].isna()]["route_id"].unique()
                    raise ValueError(f"Missing office mapping for routes: {missing_routes[:5]}")
                
                
                df = df.rename(columns={
                    "y_pred": "predicted_demand"
                })
                
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                
                
                final_cols = [
                    "office_from_id",
                    "route_id",
                    "timestamp",
                    "predicted_demand"
                ]
                df = df[final_cols]
                
                output_path = "/opt/airflow/data/solver_input.csv"
                df.to_csv(output_path, index=False)
                logger.info(f"Saved solver input with {len(df)} rows to {output_path}")
                
                return output_path
                
            except Exception as e:
                logger.error(f"Error in prepare_data: {str(e)}")
                logger.error(traceback.format_exc())
                raise
        
        prep_result = prepare_data(feature_df_data, inference_results)
        
        solve = DockerOperator(
            task_id="solve_transport_task",
            image="transport-optimizer:latest",
            api_version='auto',
            command="python /app/main.py /data/config.json /data/solver_input.csv /data/result.csv",
            mounts=[
                Mount(
                    source=os.environ["DATA_PATH"], 
                    target="/data",   
                    type="bind"
                )
            ],
            docker_url='unix://var/run/docker.sock',
            auto_remove='success',
            mount_tmp_dir=False,
            do_xcom_push=False,
            retries=0,
            user="root", # Need to fix this shit before deadline...  
        )
        
        @task
        def publish_solution():
            try:
                result_path = "/opt/airflow/data/result.csv"
                if not os.path.exists(result_path):
                    raise FileNotFoundError(f"Result file not found: {result_path}")
                
                result_df = pd.read_csv(result_path)
                logger.info(f"Solution published with {len(result_df)} rows")
                return {"status": "published", "rows": len(result_df)}
            except Exception as e:
                logger.error(f"Error in publish_solution: {str(e)}")
                logger.error(traceback.format_exc())
                raise
        
        pub = publish_solution()
        
        prep_result >> solve >> pub
        return pub 

    features = feature_engineering()
    inference = model_inference(features)
    opt = optimization(features, inference)

    features >> inference >> opt

    end = EmptyOperator(task_id="end")
    opt >> end


dag_instance = transport_pipeline()