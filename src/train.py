import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

EVAL_THRESHOLD = 0.70
LABEL_NAMES = {0: "thap", 1: "trung_binh", 2: "cao"}


def _build_model(params: dict):
    model_type = params.pop("model_type", "random_forest")
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(**params, random_state=42)
    elif model_type == "logistic_regression":
        return LogisticRegression(**params, random_state=42, max_iter=1000)
    else:
        return RandomForestClassifier(**params, random_state=42)


def _check_data_drift(y_train: pd.Series) -> dict:
    total = len(y_train)
    dist = {int(k): round(v / total, 4) for k, v in y_train.value_counts().items()}
    for cls, ratio in dist.items():
        if ratio < 0.10:
            print(f"WARNING: class {cls} ({LABEL_NAMES.get(cls, cls)}) only {ratio:.1%} of training data — possible data drift.")
    return dist


def _write_report(y_eval, preds, acc: float, f1: float) -> str:
    cm = confusion_matrix(y_eval, preds)
    report = classification_report(y_eval, preds, target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)])
    lines = [
        f"Accuracy : {acc:.4f}",
        f"F1 (weighted): {f1:.4f}",
        "",
        "Confusion Matrix:",
        str(cm),
        "",
        "Per-class Report:",
        report,
    ]
    os.makedirs("outputs", exist_ok=True)
    path = "outputs/report.txt"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: data drift check
    class_dist = _check_data_drift(y_train)

    params = dict(params)  # copy to avoid mutating caller's dict
    model_type = params.get("model_type", "random_forest")

    with mlflow.start_run():
        mlflow.log_params(params)

        # Bonus 2: multi-algorithm
        model = _build_model(params)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"Model   : {model_type}")
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # Bonus 5: include class distribution in metrics
        metrics = {"accuracy": acc, "f1_score": f1, "class_distribution": class_dist}
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Bonus 3: write performance report
        _write_report(y_eval, preds, acc, f1)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
