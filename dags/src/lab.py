import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import pickle, os, json

# ---------- Load Data ----------
def load_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    serialized_data = pickle.dumps(df)
    return serialized_data

# ---------- Preprocess Data ----------
def data_preprocessing(data):
    df = pickle.loads(data)
    df = df.dropna()
    features = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]
    X = df[features]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    serialized = pickle.dumps((X_scaled, scaler))
    return serialized

# ---------- Train KMeans ----------
def kmeans_model(data):
    X_scaled, scaler = pickle.loads(data)
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "kmeans.pkl")
    pickle.dump((kmeans, scaler), open(model_path, "wb"))
    return pickle.dumps({"model": "KMeans", "silhouette": score, "path": model_path})

# ---------- Train DBSCAN ----------
def dbscan_model(data):
    X_scaled, scaler = pickle.loads(data)
    dbscan = DBSCAN(eps=0.2, min_samples=2)
    labels = dbscan.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else -1
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "dbscan.pkl")
    pickle.dump((dbscan, scaler), open(model_path, "wb"))
    return pickle.dumps({"model": "DBSCAN", "silhouette": score, "path": model_path})

# ---------- Train Agglomerative ----------
def agglo_model(data):
    X_scaled, scaler = pickle.loads(data)
    agglo = AgglomerativeClustering(n_clusters=8)
    labels = agglo.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "agglo.pkl")
    pickle.dump((agglo, scaler), open(model_path, "wb"))
    return pickle.dumps({"model": "Agglomerative", "silhouette": score, "path": model_path})

# ---------- Evaluate Models ----------
def evaluate_models(kmeans_res, dbscan_res, agglo_res):
    results = [pickle.loads(kmeans_res), pickle.loads(dbscan_res), pickle.loads(agglo_res)]
    best = max(results, key=lambda x: x["silhouette"])
    # Save metrics
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump({"best_model": best["model"], "silhouette": best["silhouette"]}, f)
    return best["path"]

# ---------- Predict Cluster on Test Data ----------
def predict_cluster(best_model_path):
    model, scaler = pickle.load(open(best_model_path, "rb"))
    test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
    X_test = scaler.transform(test_df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]])

    # DBSCAN has no predict, so fallback to fit_predict
    if isinstance(model, DBSCAN):
        pred = model.fit_predict(X_test)
    else:
        pred = model.fit_predict(X_test) if hasattr(model, "fit_predict") else model.predict(X_test)

    print(f"Predicted cluster(s) for test data: {pred}")
    return pred.tolist()
