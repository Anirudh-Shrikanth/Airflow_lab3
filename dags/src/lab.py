import pickle
import os
import json
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ---------- Load Data ----------
def load_data():
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.3, random_state=42
    )
    data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    return pickle.dumps(data)

# ---------- Preprocess Data ----------
def preprocess_data(data):
    data_dict = pickle.loads(data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(data_dict['X_train'])
    X_test_scaled = scaler.transform(data_dict['X_test'])
    
    processed = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': data_dict['y_train'],
        'y_test': data_dict['y_test']
    }
    return pickle.dumps(processed)

# ---------- Train Logistic Regression ----------
def train_logistic(data):
    data_dict = pickle.loads(data)
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(data_dict['X_train'], data_dict['y_train'])
    
    y_pred = model.predict(data_dict['X_test'])
    accuracy = accuracy_score(data_dict['y_test'], y_pred)
    
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "logistic.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    pickle.dump(model, open(model_path, "wb"))
    
    return pickle.dumps({"model": "Logistic", "accuracy": accuracy, "path": model_path})

# ---------- Train Decision Tree ----------
def train_tree(data):
    data_dict = pickle.loads(data)
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(data_dict['X_train'], data_dict['y_train'])
    
    y_pred = model.predict(data_dict['X_test'])
    accuracy = accuracy_score(data_dict['y_test'], y_pred)
    
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "tree.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    pickle.dump(model, open(model_path, "wb"))
    
    return pickle.dumps({"model": "DecisionTree", "accuracy": accuracy, "path": model_path})

# ---------- Train KNN ----------
def train_knn(data):
    data_dict = pickle.loads(data)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(data_dict['X_train'], data_dict['y_train'])
    
    y_pred = model.predict(data_dict['X_test'])
    accuracy = accuracy_score(data_dict['y_test'], y_pred)
    
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "knn.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    pickle.dump(model, open(model_path, "wb"))
    
    return pickle.dumps({"model": "KNN", "accuracy": accuracy, "path": model_path})

# ---------- Evaluate Models ----------
def evaluate_models(logistic_res, tree_res, knn_res):
    results = [pickle.loads(logistic_res), pickle.loads(tree_res), pickle.loads(knn_res)]
    best = max(results, key=lambda x: x["accuracy"])
    
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump({
            "best_model": best["model"],
            "accuracy": best["accuracy"],
            "all_results": results
        }, f, indent=2)
    
    print(f"Best model: {best['model']} with accuracy: {best['accuracy']:.4f}")
    return best["path"]

# ---------- Make Predictions ----------
def predict(best_model_path):
    data_dict = pickle.loads(load_data())
    model = pickle.load(open(best_model_path, "rb"))
    
    predictions = model.predict(data_dict['X_test'][:5])
    actuals = data_dict['y_test'][:5]
    
    print(f"Predictions: {predictions}")
    print(f"Actuals: {actuals}")
    return predictions.tolist()