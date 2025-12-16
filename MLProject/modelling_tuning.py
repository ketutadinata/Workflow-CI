# MLProject/modelling.py (Adaptasi dari modelling_tuning.py)

import os
import pandas as pd
import mlflow
import mlflow.sklearn
import argparse
import joblib
# Import semua library yang Anda gunakan untuk training & logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

# Inisialisasi Argumen dari MLProject
def parse_args():
    parser = argparse.ArgumentParser()
    # Tambahkan hyperparameter yang ingin di-tuning/log
    parser.add_argument("--rf_n_estimators", type=int, default=100)
    parser.add_argument("--logreg_C", type=float, default=1.0) 
    return parser.parse_args()

# Fungsi utama yang akan dijalankan oleh MLflow Project
def main():
    args = parse_args()
    
    # ----------------------------
    # INIT DAGSHUB + MLFLOW (SESUAIKAN DENGAN REPO ANDA)
    # ----------------------------
    dagshub.init(
        repo_owner='<nama_owner_anda>',
        repo_name='<nama_repo_dagshub_anda>', 
        mlflow=True
    )

    # ----------------------------
    # KONSTANTA DATASET (Asumsikan dataset sudah ada di folder yang sama)
    # ----------------------------
    # Ubah path ini agar sesuai dengan struktur MLProject/
    TRAIN_PATH = "namadataset_preprocessing/train_preprocessed.csv"
    TEST_PATH = "namadataset_preprocessing/test_preprocessed.csv"
    TARGET_COL = "Personality"
    
    # Pastikan folder artifacts ada
    os.makedirs("artifacts", exist_ok=True)
    
    # LOAD DATA (Logika sama seperti di modelling_tuning.py)
    # ... (code loading data)
    
    # ----------------------------
    # MLFLOW RUN
    # ----------------------------
    # Nama eksperimen harus spesifik
    with mlflow.start_run(run_name="CI_Retraining_Run"):
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        
        # 1. MODEL 1: Random Forest + Tuning (Gunakan args dari MLProject)
        param_grid_rf = {'n_estimators': [args.rf_n_estimators], 'max_depth': [5, 10]}
        # ... (code GridSearchCV untuk RF)
        
        # 2. MODEL 2: Logistic Regression + Tuning (Gunakan args dari MLProject)
        param_grid_logreg = {'C': [args.logreg_C, 10.0]}
        # ... (code GridSearchCV untuk LogReg)

        # 3. PILIH BEST MODEL (Logika sama)
        # ... (code pemilihan best_model)
        
        # 4. MANUAL LOGGING (Sangat Penting untuk Kriteria 2 Advanced)
        # Log params & metrics (menggunakan manual logging)
        # ... (Logika logging metrics, params, dan model)

        # 5. LOG ARTIFACT TAMBAHAN (Wajib Kriteria 2 Advanced)
        # ... (Logika logging Confusion Matrix, Feature Importance, Dataset Snapshot)

        # Simpan Model ke MLflow
        mlflow.sklearn.log_model(
            sk_model=best_model, 
            artifact_path="model", 
            registered_model_name="PersonalityClassifier"
        )
        print("Model, metrics, dan artifacts telah dilog ke DagsHub.")

if __name__ == "__main__":
    main()