#!/usr/bin/env python3
import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ======================
# CONFIG
# ======================
ARTIFACT_DIR = "artifacts"
DATA_PATH = os.path.join(ARTIFACT_DIR, "train_preprocessed.csv")

print("=" * 60)
print("Starting MLflow Training (MLflow Project Mode)")
print("=" * 60)

# ======================
# LOAD DATA
# ======================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

X = df.drop("Personality", axis=1)
y = df["Personality"]

# ======================
# SPLIT DATA
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train size: {len(X_train)}")
print(f"Test size : {len(X_test)}")

# ======================
# SCALING
# ======================
os.makedirs(ARTIFACT_DIR, exist_ok=True)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler_path = os.path.join(ARTIFACT_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")

# ======================
# MODEL TUNING
# ======================
print("\nHyperparameter tuning...")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

print(f"Best params    : {grid_search.best_params_}")
print(f"Best CV score  : {grid_search.best_score_:.4f}")

# ======================
# EVALUATION
# ======================
y_pred = best_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("\nEvaluation on test set:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

# ======================
# MLFLOW LOGGING
# ======================
print("\nLogging artifacts and metrics to MLflow...")

# Parameter logging
mlflow.log_params(grid_search.best_params_)
mlflow.log_param("model_type", "RandomForestClassifier")
mlflow.log_param("cv_folds", 3)

# Metrics logging
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("precision_weighted", precision)
mlflow.log_metric("recall_weighted", recall)
mlflow.log_metric("f1_score_weighted", f1)
mlflow.log_metric("best_cv_score", grid_search.best_score_)

# Model logging
mlflow.sklearn.log_model(
    sk_model=best_model,
    artifact_path="model"
)

# Artifact logging
mlflow.log_artifact(scaler_path, artifact_path="preprocessing")

# Save model locally for CI verification
model_path = os.path.join(ARTIFACT_DIR, "best_model.pkl")
joblib.dump(best_model, model_path)
print(f"Model saved to {model_path}")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)
