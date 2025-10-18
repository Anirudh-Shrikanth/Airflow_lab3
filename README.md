This repo contains all materials for Lab1 of Airflow_Labs.

The following are my changes:
# Multi-Model Clustering with Apache Airflow

This project demonstrates an **machine learning pipeline** orchestrated using **Apache Airflow**.  
It trains multiple clustering models — **KMeans**, **DBSCAN**, and **Agglomerative Clustering** — on a credit card dataset, evaluates their performance, selects the best model automatically, and uses it to predict clusters for test data.

## Features

- **Automated data preprocessing** using `MinMaxScaler`
- **Multiple clustering models**:
  - KMeans
  - DBSCAN
  - Agglomerative Clustering
- **Automatic model selection** using Silhouette Score
- **Cluster prediction** on new (test) data
- **Full orchestration** through Apache Airflow DAGs
- **Modular structure** — easy to extend or replace models

---

## Setup Instructions

### Prerequisites
- **Docker** and **Docker Compose** installed on your system
- Python is **not required locally**, as all dependencies are handled within Docker

---

### Start Airflow
Run the following from the project root:

```bash
docker-compose up -d
```

This will:
- Build and start Airflow containers (webserver, scheduler, Postgres)
- Install the required Python libraries

If you modify dependencies, update the following line in `docker-compose.yaml`:

```yaml
_PIP_ADDITIONAL_REQUIREMENTS: pandas,scikit-learn,kneed
```

Then rebuild the containers:

```bash
docker-compose down
docker-compose up --build -d
```

---

### Access the Airflow UI

Open [http://localhost:8080](http://localhost:8080) in your browser.  
- **Username:** airflow  
- **Password:** airflow  

---

## DAG Overview

**DAG ID:** `Multi_Model_Clustering`

### Tasks:

| Task ID | Description |
|----------|--------------|
| `load_data_task` | Loads data from `data/file.csv` |
| `data_preprocessing_task` | Cleans data, scales features using MinMaxScaler |
| `kmeans_model_task` | Trains a KMeans clustering model |
| `dbscan_model_task` | Trains a DBSCAN clustering model |
| `agglo_model_task` | Trains an Agglomerative Clustering model |
| `evaluate_models_task` | Compares models via S

![Description of image](screenshots/homepage.png)
![Description of image](screenshots/homepage.png)