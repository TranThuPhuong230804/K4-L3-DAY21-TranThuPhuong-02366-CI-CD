import json
import os

import numpy as np
import pandas as pd

from src.train import train


FEATURE_NAMES = [
    "age",
    "workclass",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]


def _make_temp_data(tmp_path):
    """Create a small deterministic dataset with the Adult feature schema."""
    rng = np.random.default_rng(0)
    n_samples = 200
    features = rng.random((n_samples, len(FEATURE_NAMES)))
    target = rng.integers(0, 2, size=n_samples)

    df = pd.DataFrame(features, columns=FEATURE_NAMES)
    df["target"] = target

    train_path = str(tmp_path / "train.csv")
    eval_path = str(tmp_path / "holdout.csv")
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)
    return train_path, eval_path


def test_train_returns_float(tmp_path, monkeypatch):
    """The training function returns an F1 score in the valid range."""
    monkeypatch.chdir(tmp_path)
    train_path, eval_path = _make_temp_data(tmp_path)

    f1 = train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert isinstance(f1, float)
    assert 0.0 <= f1 <= 1.0


def test_report_file_created(tmp_path, monkeypatch):
    """Training writes both expected metrics to outputs/report.json."""
    monkeypatch.chdir(tmp_path)
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )

    report_path = tmp_path / "outputs" / "report.json"
    assert report_path.exists()
    with report_path.open(encoding="utf-8") as report_file:
        report = json.load(report_file)
    assert "f1_score" in report
    assert "accuracy" in report
    assert 0.0 <= report["f1_score"] <= 1.0
    assert 0.0 <= report["accuracy"] <= 1.0


def test_model_file_created(tmp_path, monkeypatch):
    """Training saves the fitted model under models/model.joblib."""
    monkeypatch.chdir(tmp_path)
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists(tmp_path / "models" / "model.joblib")
