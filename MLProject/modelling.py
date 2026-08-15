"""
modelling.py (Workflow-CI / MLProject)
Dipanggil oleh `mlflow run` di dalam GitHub Actions.
Melatih model dan mencatat run ke MLflow Tracking lokal (./mlruns pada runner),
yang kemudian diupload sebagai artefak CI dan dipakai untuk build Docker image.
"""

import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

TARGET_COL = "Churn"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="telco_preprocessing/train.csv")
    parser.add_argument("--test_path", default="telco_preprocessing/test.csv")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=20)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    with mlflow.start_run(run_name="ci_random_forest"):
        mlflow.log_params(
            {"n_estimators": args.n_estimators, "max_depth": args.max_depth}
        )

        model = RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

        mlflow.sklearn.log_model(
            model,
            "model",
            pip_requirements=[
                "mlflow==2.19.0",
                "scikit-learn==1.5.2",
                "pandas==2.2.3",
                "pyarrow==16.1.0",  
                ],
        )
        print("CI training run selesai.")


if __name__ == "__main__":
    main()
