"""Reproducible customer-churn analysis for the Telco CSV data set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier


TARGET = "Churn"
DROP_COLUMNS = ["customerID"]


def prepare_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean the Telco columns and return model features and a binary target."""
    data = frame.copy()
    if TARGET not in data:
        raise ValueError(f"Input data must contain a {TARGET!r} column.")
    data = data.drop(columns=[column for column in DROP_COLUMNS if column in data])
    if "TotalCharges" in data:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    target = data.pop(TARGET)
    if target.dtype == "object":
        target = target.astype(str).str.strip().map({"Yes": 1, "No": 0})
    target = pd.to_numeric(target, errors="coerce")
    if target.isna().any():
        raise ValueError("Churn must contain only Yes/No (or 0/1) values.")
    return data, target.astype(int)


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = features.select_dtypes(include=["number"]).columns.tolist()
    categorical = features.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)]
    )


def make_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    """Create comparable pipelines; preprocessing is fitted only on training data."""
    return {
        "Logistic regression": Pipeline(
            [("preprocess", preprocessor), ("model", LogisticRegression(max_iter=1000, random_state=42))]
        ),
        "k-nearest neighbours": Pipeline(
            [("preprocess", preprocessor), ("model", KNeighborsClassifier(n_neighbors=15))]
        ),
        "Decision tree": Pipeline(
            [
                ("preprocess", preprocessor),
                ("model", DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)),
            ]
        ),
        "Random forest": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_models(
    models: dict[str, Pipeline],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict, dict[str, Pipeline]]:
    metrics = {}
    fitted = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics[name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        }
        fitted[name] = model
    return metrics, fitted


def make_visualizations(
    test_features: pd.DataFrame,
    y_test: pd.Series,
    fitted: dict[str, Pipeline],
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    y_test.value_counts().sort_index().rename({0: "Stayed", 1: "Churned"}).plot.bar(ax=ax)
    ax.set_title("Test-set class distribution")
    ax.set_ylabel("Customers")
    fig.tight_layout()
    fig.savefig(output / "class_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in fitted.items():
        RocCurveDisplay.from_estimator(model, test_features, y_test, name=name, ax=ax)
    ax.set_title("ROC curves (held-out test set)")
    fig.tight_layout()
    fig.savefig(output / "roc_curves.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for axis, (name, model) in zip(axes.ravel(), fitted.items()):
        ConfusionMatrixDisplay.from_estimator(
            model, test_features, y_test, display_labels=["Stayed", "Churned"], ax=axis
        )
        axis.set_title(name)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrices.png", dpi=160)
    plt.close(fig)

    tree = fitted["Decision tree"]
    names = tree.named_steps["preprocess"].get_feature_names_out()
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(
        tree.named_steps["model"],
        feature_names=names,
        class_names=["Stayed", "Churned"],
        filled=True,
        max_depth=3,
        fontsize=7,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(output / "decision_tree.png", dpi=160)
    plt.close(fig)


def run(input_csv: str | Path, output_dir: str | Path = "outputs", random_state: int = 42) -> dict:
    frame = pd.read_csv(input_csv)
    features, target = prepare_data(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=random_state
    )
    models = make_models(make_preprocessor(features))
    metrics, fitted = evaluate_models(models, x_train, x_test, y_train, y_test)
    output = Path(output_dir)
    make_visualizations(x_test, y_test, fitted, output)
    results = {
        "rows": len(frame),
        "test_rows": len(x_test),
        "random_state": random_state,
        "metrics": metrics,
    }
    (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", help="Path to WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("-o", "--output-dir", default="outputs")
    args = parser.parse_args()
    results = run(args.input_csv, args.output_dir)
    print(json.dumps(results["metrics"], indent=2))


if __name__ == "__main__":
    main()
