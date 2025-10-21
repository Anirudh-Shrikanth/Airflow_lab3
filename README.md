# Wine Classification Pipeline

A simple Apache Airflow pipeline that trains multiple machine learning models on the Wine dataset and selects the best performing model.

## Overview

This project demonstrates a complete MLOps pipeline using Airflow for orchestration:
- Loads Wine dataset (178 samples, 13 features, 3 classes)
- Preprocesses data with StandardScaler
- Trains 3 models in parallel: Logistic Regression, Decision Tree, and KNN
- Evaluates and selects best model based on accuracy
- Makes predictions on test data

## Requirements
```bash
_PIP_ADDITIONAL_REQUIREMENTS: ${_PIP_ADDITIONAL_REQUIREMENTS:- pandas scikit-learn kneed }
```

## Quick Start

1. **Start Airflow**
```bash
   docker-compose up
```

2. **Access UI**
   - Navigate to `http://localhost:8080`
   - Default credentials: admin/admin
   - Enable and trigger the `wine_classification` DAG

## Pipeline Flow
```
load_data
    ↓
preprocess_data
    ↓
    ├─→ train_logistic ─┐
    ├─→ train_tree ─────┼─→ evaluate_models → predict
    └─→ train_knn ──────┘
```

### Tasks

1. **load_data**: Loads Wine dataset and splits into train/test (70/30)
2. **preprocess_data**: Applies StandardScaler to features
3. **train_logistic**: Trains Logistic Regression
4. **train_tree**: Trains Decision Tree (max_depth=5)
5. **train_knn**: Trains K-Nearest Neighbors (k=5)
6. **evaluate_models**: Compares accuracies and selects best model
7. **predict**: Makes predictions on test data using best model

## Output

After successful execution:

**`results/results.json`**
```json
{
  "best_model": "Logistic",
  "accuracy": 0.9815,
  "all_results": [
    {"model": "Logistic", "accuracy": 0.9815, "path": "..."},
    {"model": "DecisionTree", "accuracy": 0.9259, "path": "..."},
    {"model": "KNN", "accuracy": 0.9630, "path": "..."}
  ]
}
```

**`model/`** directory contains:
- `logistic.pkl`
- `tree.pkl`
- `knn.pkl`

### Add More Models

1. Add training function in `lab.py`:
```python
   def train_svm(data):
       from sklearn.svm import SVC
       data_dict = pickle.loads(data)
       model = SVC(random_state=42)
```

2. Add task in `airflow.py`:
```python
   svm_task = PythonOperator(
       task_id="train_svm",
       python_callable=train_svm,
       op_args=[preprocess_task.output],
   )
```

3. Update dependencies:
```python
   preprocess_task >> [logistic_task, tree_task, knn_task, svm_task]
   [logistic_task, tree_task, knn_task, svm_task] >> evaluate_task
```

### Modify Hyperparameters

Edit model parameters in training functions:
```python
# In train_logistic()
model = LogisticRegression(random_state=42, max_iter=200)

model = LogisticRegression(random_state=42, max_iter=500, C=0.1)

# In train_tree()
model = DecisionTreeClassifier(random_state=42, max_depth=5)

model = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5)

# In train_knn()
model = KNeighborsClassifier(n_neighbors=5)

model = KNeighborsClassifier(n_neighbors=3, weights='distance')
```

## Monitoring

- **Airflow UI**: Monitor task execution, logs, and XCom data
- **Task Logs**: Click on task -> View Logs to see print statements
- **Results File**: Check `results/results.json` for model comparison

