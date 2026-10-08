import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# The release quality gate is based on the positive-class F1 score.
F1_THRESHOLD = 0.65
TARGET_COLUMN = "target"


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """Train a model, log its metrics to MLflow, and save release artifacts.

    Returns the F1 score for target=1 (annual income greater than $50K) on
    the holdout dataset.
    """
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    if TARGET_COLUMN not in df_train or TARGET_COLUMN not in df_eval:
        raise ValueError(f"Both datasets must contain a '{TARGET_COLUMN}' column")
    if df_train.empty or df_eval.empty:
        raise ValueError("Training and evaluation datasets must not be empty")

    X_train = df_train.drop(columns=[TARGET_COLUMN])
    y_train = df_train[TARGET_COLUMN]
    X_eval = df_eval.drop(columns=[TARGET_COLUMN])
    y_eval = df_eval[TARGET_COLUMN]

    if list(X_train.columns) != list(X_eval.columns):
        raise ValueError("Training and evaluation datasets must have matching features")

    with mlflow.start_run():
        mlflow.log_params(params)

        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_eval)
        f1 = float(f1_score(y_eval, predictions, zero_division=0))
        accuracy = float(accuracy_score(y_eval, predictions))

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"F1: {f1:.4f} | Accuracy: {accuracy:.4f}")

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.json", "w", encoding="utf-8") as report_file:
            json.dump({"f1_score": f1, "accuracy": accuracy}, report_file, indent=2)
            report_file.write("\n")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    return f1


if __name__ == "__main__":
    with open("params.yaml", encoding="utf-8") as params_file:
        params = yaml.safe_load(params_file)
    train(params)
