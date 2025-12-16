import os
import argparse
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import dagshub.dagshub as dh

# ----------------------------
# ARGUMENT PARSER
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf_n_estimators", type=int, default=200)
    parser.add_argument("--logreg_C", type=float, default=1.0)
    return parser.parse_args()

# ----------------------------
# MAIN FUNCTION
# ----------------------------
def main():
    args = parse_args()

    # ----------------------------
    # INIT DAGSHUB + MLFLOW
    # ----------------------------
    dh.init(
        repo_owner="ketutadinata1811",
        repo_name="my-first-repo",
        mlflow=True,
        token=os.environ.get("DAGSHUB_TOKEN")
    )

    mlflow.set_experiment("eksperimen-mlflow")

    # ----------------------------
    # DATASET PATH
    # ----------------------------
    TRAIN_PATH = "namadataset_preprocessing/train_preprocessed.csv"
    TEST_PATH = "namadataset_preprocessing/test_preprocessed.csv"
    TARGET_COL = "Personality"

    os.makedirs("artifacts", exist_ok=True)

    # ----------------------------
    # LOAD DATA
    # ----------------------------
    import pandas as pd
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
        mlflow.log_metric("rf_f1_macro", rf_f1)
        mlflow.sklearn.log_model(rf, "rf_model")

        # ---------- LOGISTIC REGRESSION ----------
        logreg = LogisticRegression(C=args.logreg_C, max_iter=1000)
        logreg.fit(X_train, y_train)
        lr_pred = logreg.predict(X_test)
        lr_f1 = f1_score(y_test, lr_pred, average="macro")
        mlflow.log_metric("logreg_f1_macro", lr_f1)
        mlflow.sklearn.log_model(logreg, "logreg_model")

        # ---------- BEST MODEL ----------
        best_model = rf if rf_f1 > lr_f1 else logreg
        best_name = "RandomForest" if rf_f1 > lr_f1 else "LogisticRegression"
        joblib.dump(best_model, "artifacts/best_model.pkl")
        mlflow.log_param("best_model", best_name)
        mlflow.log_artifact("artifacts/best_model.pkl")

        print(f"Best model: {best_name}, RF F1: {rf_f1}, LR F1: {lr_f1}")
        print("RUN SELESAI (CI-Friendly Version)")

if __name__ == "__main__":
    main()
