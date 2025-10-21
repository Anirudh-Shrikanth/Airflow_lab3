from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow import configuration as conf
from src.lab import (
    load_data, preprocess_data,
    train_logistic, train_tree, train_knn,
    evaluate_models, predict
)

# Enable XCom pickling
conf.set("core", "enable_xcom_pickling", "True")

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 1, 15),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "wine_classification",
    default_args=default_args,
    description="Simple wine classification with 3 models",
    schedule_interval=None,
    catchup=False,
    tags=["classification", "wine"],
) as dag:

    load_task = PythonOperator(
        task_id="load_data",
        python_callable=load_data,
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
        op_args=[load_task.output],
    )

    logistic_task = PythonOperator(
        task_id="train_logistic",
        python_callable=train_logistic,
        op_args=[preprocess_task.output],
    )

    tree_task = PythonOperator(
        task_id="train_tree",
        python_callable=train_tree,
        op_args=[preprocess_task.output],
    )

    knn_task = PythonOperator(
        task_id="train_knn",
        python_callable=train_knn,
        op_args=[preprocess_task.output],
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_models",
        python_callable=evaluate_models,
        op_args=[logistic_task.output, tree_task.output, knn_task.output],
    )

    predict_task = PythonOperator(
        task_id="predict",
        python_callable=predict,
        op_args=[evaluate_task.output],
    )

    # Dependencies
    load_task >> preprocess_task >> [logistic_task, tree_task, knn_task] >> evaluate_task >> predict_task