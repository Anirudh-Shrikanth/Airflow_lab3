from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from airflow import configuration as conf
from src.lab import (
    load_data, data_preprocessing,
    kmeans_model, dbscan_model, agglo_model,
    evaluate_models, predict_cluster
)

# Enable XCom pickling
conf.set("core", "enable_xcom_pickling", "True")

default_args = {
    "owner": "your_name",
    "start_date": datetime(2025, 1, 15),
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "Multi_Model_Clustering",
    default_args=default_args,
    description="Train multiple clustering models and pick the best",
    schedule_interval=None,
    catchup=False,
)

# -------- DAG tasks --------
load_data_task = PythonOperator(
    task_id="load_data_task",
    python_callable=load_data,
    dag=dag,
)

data_preprocessing_task = PythonOperator(
    task_id="data_preprocessing_task",
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)

kmeans_task = PythonOperator(
    task_id="kmeans_model_task",
    python_callable=kmeans_model,
    op_args=[data_preprocessing_task.output],
    dag=dag,
)

dbscan_task = PythonOperator(
    task_id="dbscan_model_task",
    python_callable=dbscan_model,
    op_args=[data_preprocessing_task.output],
    dag=dag,
)

agglo_task = PythonOperator(
    task_id="agglo_model_task",
    python_callable=agglo_model,
    op_args=[data_preprocessing_task.output],
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id="evaluate_models_task",
    python_callable=evaluate_models,
    op_args=[kmeans_task.output, dbscan_task.output, agglo_task.output],
    dag=dag,
)

predict_task = PythonOperator(
    task_id="predict_cluster_task",
    python_callable=predict_cluster,
    op_args=[evaluate_task.output],
    dag=dag,
)

# -------- Set dependencies --------
load_data_task >> data_preprocessing_task
data_preprocessing_task >> [kmeans_task, dbscan_task, agglo_task]
[kmeans_task, dbscan_task, agglo_task] >> evaluate_task >> predict_task

if __name__ == "__main__":
    dag.cli()
