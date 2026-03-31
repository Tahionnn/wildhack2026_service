from datetime import datetime, timedelta

import logging

from airflow.decorators import dag, task_group
from airflow.providers.standard.operators.empty import EmptyOperator


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
        extract = EmptyOperator(task_id="extract_raw_data")
        make = EmptyOperator(task_id="make_featurecs")
        extract >> make 

    @task_group(group_id="inference")
    def model_inference():
        prep = EmptyOperator(task_id="prepare_inference_request")
        run = EmptyOperator(task_id="run_model_inference")
        prep >> run 

    @task_group(group_id="optimization")
    def optimization():
        prep = EmptyOperator(task_id="prepare_optimization_task")
        solve = EmptyOperator(task_id="solve_transport_task")
        pub = EmptyOperator(task_id="publish_solution")
        prep >> solve >> pub 

    features = feature_engineering()
    inference = model_inference()
    opt = optimization()

    features >> inference >> opt

    end = EmptyOperator(task_id="end")
    opt >> end


dag_instance =  transport_pipeline()