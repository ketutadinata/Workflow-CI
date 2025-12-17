import os
import argparse
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# ----------------------------
# KONFIGURASI PATH (SINKRONISASI)
# ----------------------------
# Jalur absolut ke root folder dari MLProject/modelling_tuning.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Target folder artifacts di tingkat root
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf_n_estimators", type=int, default=200)
    parser.add_argument("--logreg_C", type=float, default=1.0)
    return parser.parse_args()

def main():
    args = parse_args()

    # Pastikan folder artifacts ada di root
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # ----------------------------
    # MLFLOW SETUP (LOKAL)
    # ----------------------------
    mlflow.set_experiment("eksperimen-mlflow")

    # ----------------------------
    # DATASET PATH
    # ----------------------------
    # Mengambil dataset dari root/artifacts/
    TRAIN_PATH = os.path.join(ARTIFACT_DIR, "train_preprocessed.csv")
    TEST_PATH = os.path.join(ARTIFACT_DIR, "test_preprocessed.csv")
    TARGET_COL = "Personality"

    # Cek apakah dataset ada sebelum lanjut
    if not os.path.exists(TRAIN_PATH):
        print(f"❌ Error: File {TRAIN_PATH} tidak ditemukan!")
        print(f"Cek folder: {ARTIFACT_DIR}")
        return

    # ----------------------------
    # LOAD DATA
    # ----------------------------
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(TARGET_COL, axis=1)
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(TARGET_COL, axis=1)
    y_test = test_df[TARGET_COL]

    # ----------------------------
    # MLflow RUN
    # ----------------------------
    with mlflow.start_run(run_name="CI_Retraining_Run"):

        # ---------- RANDOM FOREST ----------
        rf = RandomForestClassifier(n_estimators=args.rf_n_estimators, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_f1 = f1_score(y_test, rf_pred, average="macro")
        
        mlflow.log_param("rf_n_estimators", args.rf_n_estimators)
        mlflow.log_metric("rf_f1_macro", rf_f1)

        # ---------- LOGISTIC REGRESSION ----------
        logreg = LogisticRegression(C=args.logreg_C, max_iter=1000)
        logreg.fit(X_train, y_train)
        lr_pred = logreg.predict(X_test)
        lr_f1 = f1_score(y_test, lr_pred, average="macro")
        
        mlflow.log_param("logreg_C", args.logreg_C)
        mlflow.log_metric("logreg_f1_macro", lr_f1)

        # ---------- BEST MODEL SELECTION ----------
        # Variabel best_model didefinisikan di sini
        if rf_f1 > lr_f1:
            best_model = rf
            best_name = "RandomForest"
        else:
            best_model = logreg
            best_name = "LogisticRegression"
        
        # Simpan ke root/artifacts/best_model.pkl
        model_save_path = os.path.join(ARTIFACT_DIR, "best_model.pkl")
        joblib.dump(best_model, model_save_path)
        
        mlflow.log_param("best_model_selected", best_name)
        mlflow.sklearn.log_model(best_model, "best_model")
        mlflow.log_artifact(model_save_path)

        print(f"✅ Berhasil! Best model: {best_name}")
        print(f"✅ RF F1: {rf_f1:.4f} | LR F1: {lr_f1:.4f}")
        print(f"✅ Model disimpan di: {model_save_path}")

if __name__ == "__main__":
    main()