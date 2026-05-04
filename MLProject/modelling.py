"""
modelling_tuning.py
Hyperparameter tuning RandomForest dengan manual MLflow logging + DagsHub (Advanced).

Usage:
    python modelling_tuning.py
    python modelling_tuning.py --use_dagshub True
"""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import joblib
import random
import numpy as np
import pandas as pd
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix)

SEED            = 42
EXPERIMENT_NAME = "Wildfire_Classification_Tuning"
LABEL_MAP       = {0: "Low", 1: "Moderate", 2: "High", 3: "Extreme"}
np.random.seed(SEED)
random.seed(SEED)


def setup_dagshub(tracking_uri: str, username: str, password: str):
    os.environ["MLFLOW_TRACKING_URI"]      = tracking_uri
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password
    mlflow.set_tracking_uri(tracking_uri)
    print(f"✔ MLflow → DagsHub: {tracking_uri}")


def load_data(data_dir: str):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val   = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "test.csv"))

    target = "intensity_class"
    # Gabungkan train+val untuk GridSearchCV dengan CV internal
    train_full = pd.concat([train, val], ignore_index=True)

    X_train = train_full.drop(columns=[target]).values
    y_train = train_full[target].values.astype(int)
    X_test  = test.drop(columns=[target]).values
    y_test  = test[target].values.astype(int)
    feature_cols = list(test.drop(columns=[target]).columns)

    print(f"Train+Val: {X_train.shape} | Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test, feature_cols


def save_confusion_matrix(y_true, y_pred, path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_MAP.values(),
                yticklabels=LABEL_MAP.values())
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def save_feature_importance(model, feature_cols, path="feature_importance.png"):
    importances = model.best_estimator_.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_cols) * 0.4)))
    ax.barh(fi_df["feature"], fi_df["importance"], color="#3498DB", edgecolor="black")
    ax.set_title("Feature Importance (Best Model)", fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def save_per_class_metrics(y_true, y_pred, path="per_class_metrics.json"):
    report = classification_report(
        y_true, y_pred,
        target_names=list(LABEL_MAP.values()),
        output_dict=True, zero_division=0
    )
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


def main(args):
    if args.use_dagshub:
        uri  = os.environ.get("MLFLOW_TRACKING_URI", "")
        user = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
        pwd  = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
        if not uri:
            raise ValueError("Set MLFLOW_TRACKING_URI ke DagsHub URI kamu.")
        setup_dagshub(uri, user, pwd)

    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, y_train, X_test, y_test, feature_cols = load_data(args.data_dir)

    # ── Hyperparameter Grid ───────────────────────────────────────────────────
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth":    [10, 20, None],
        "min_samples_split": [5, 10],
    }
    # Total combinations: 2 × 3 × 2 = 12, dengan CV=3 → 36 fits
    # Dengan n_jobs=-1 selesai ~5-10 menit

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    base_model = RandomForestClassifier(
        class_weight="balanced", random_state=SEED, n_jobs=-1
    )
    grid_search = GridSearchCV(
        base_model, param_grid,
        cv=cv, scoring="f1_weighted",
        n_jobs=-1, verbose=2, refit=True
    )

    print("\nMenjalankan GridSearchCV...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    print(f"\nBest params: {best_params}")
    print(f"Best CV F1 (weighted): {best_cv_score:.4f}")

    # ── Evaluasi pada test set ─────────────────────────────────────────────────
    y_pred = best_model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"Test F1 (wtd) : {f1:.4f}")
    print(classification_report(
        y_test, y_pred,
        target_names=list(LABEL_MAP.values()), zero_division=0))

    # ── Manual MLflow Logging ─────────────────────────────────────────────────
    with mlflow.start_run(run_name=f"RF_GridSearch_best") as run:
        # Parameters
        mlflow.log_params({
            "model_type":        "RandomForest",
            "n_estimators":      best_params["n_estimators"],
            "max_depth":         str(best_params["max_depth"]),
            "min_samples_split": best_params["min_samples_split"],
            "class_weight":      "balanced",
            "cv_folds":          3,
            "cv_scoring":        "f1_weighted",
            "seed":              SEED,
        })

        # Metrics utama
        mlflow.log_metrics({
            "best_cv_f1_weighted": best_cv_score,
            "test_accuracy":       acc,
            "test_f1_weighted":    f1,
            "test_precision":      prec,
            "test_recall":         rec,
        })

        # Per-class metrics (manual — tidak ada di autolog)
        for cls_idx, cls_name in LABEL_MAP.items():
            mlflow.log_metric(
                f"test_f1_class_{cls_name.lower()}",
                float(f1_per_class[cls_idx])
            )

        # Artefak tambahan 1: Confusion Matrix
        cm_path = save_confusion_matrix(y_test, y_pred)
        mlflow.log_artifact(cm_path, "plots")

        # Artefak tambahan 2: Feature Importance
        fi_path = save_feature_importance(grid_search, feature_cols)
        mlflow.log_artifact(fi_path, "plots")

        # Artefak tambahan 3: Per-class metrics JSON
        cls_path = save_per_class_metrics(y_test, y_pred)
        mlflow.log_artifact(cls_path, "metrics")

        # Simpan model
        mlflow.sklearn.log_model(
            best_model, "model",
            registered_model_name="WildfireClassifier_Tuned"
        )

        # Simpan run_id
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)
        mlflow.log_artifact("run_id.txt")

        print(f"\n✔ Run ID: {run.info.run_id}")
        print("Cek di MLflow UI atau DagsHub.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    type=str,  default="wildfire_preprocessing")
    parser.add_argument("--use_dagshub", action="store_true")
    args = parser.parse_args()
    main(args)