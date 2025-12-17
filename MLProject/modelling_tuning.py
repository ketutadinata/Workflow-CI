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

# Config
ARTIFACT_DIR = "artifacts"
DATA_PATH = os.path.join(ARTIFACT_DIR, "train_preprocessed.csv")
EXPERIMENT_NAME = "Personality_Classification_Tuning"

print("="*60)
print("Starting MLflow Training")
print("="*60)

# Setup MLflow
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'file:///tmp/mlruns'))
mlflow.set_experiment(EXPERIMENT_NAME)

# Load data
print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")

X = df.drop("Personality", axis=1)
y = df["Personality"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
scaler_path = os.path.join(ARTIFACT_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"Scaler saved: {scaler_path}")

# Hyperparameter tuning
print("\nHyperparameter tuning...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=3, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Evaluate
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# MLflow logging
print("\nLogging to MLflow...")
with mlflow.start_run(run_name="RandomForest_Tuned"):
    # Parameters
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("cv_folds", 3)
    
    # Metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_weighted", precision)
    mlflow.log_metric("recall_weighted", recall)
    mlflow.log_metric("f1_score_weighted", f1)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    
    # Model
    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model",
        registered_model_name="RandomForest_Personality_Classifier"
    )
    
    # Scaler artifact
    mlflow.log_artifact(scaler_path, artifact_path="preprocessing")
    
    # Save locally
    model_path = os.path.join(ARTIFACT_DIR, "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"Model saved: {model_path}")

print("\n" + "="*60)
print("✅ TRAINING COMPLETED!")
print("="*60)