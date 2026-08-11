"""
Online Shoppers Purchasing Intention — classification training script.

Run from project root or from model/:
  python model/online_shoppers_classification.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
CWD = Path.cwd()
if (CWD / "app.py").exists() or (CWD / "data").exists():
    ROOT = CWD
elif SCRIPT_DIR.name == "model" and (SCRIPT_DIR.parent / "app.py").exists():
    ROOT = SCRIPT_DIR.parent
elif (CWD.parent / "app.py").exists():
    ROOT = CWD.parent
else:
    ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "model" else CWD

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/"
    "online_shoppers_intention.csv"
)
LOCAL_CSV = DATA_DIR / "online_shoppers_intention.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20

NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]
CATEGORICAL_FEATURES = ["Month", "VisitorType", "Weekend"]
TARGET_COL = "Revenue"
FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_one_hot() -> OneHotEncoder:
    params = {"handle_unknown": "ignore"}
    if "sparse_output" in OneHotEncoder.__init__.__code__.co_varnames:
        params["sparse_output"] = False
    else:
        params["sparse"] = False
    return OneHotEncoder(**params)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", make_one_hot(), CATEGORICAL_FEATURES),
        ]
    )


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "AUC": round(float(roc_auc_score(y_true, y_proba)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 4),
    }


def load_dataset() -> pd.DataFrame:
    if LOCAL_CSV.exists():
        print("Local file:", LOCAL_CSV)
        return pd.read_csv(LOCAL_CSV)

    print("Downloading from UCI...")
    try:
        df = pd.read_csv(DATASET_URL)
        df.to_csv(LOCAL_CSV, index=False)
        return df
    except Exception as err:
        import ssl
        from urllib.request import urlopen

        print("Retry download:", err)
        ctx = ssl._create_unverified_context()
        with urlopen(DATASET_URL, context=ctx) as resp:
            LOCAL_CSV.write_bytes(resp.read())
        return pd.read_csv(LOCAL_CSV)


def main() -> None:
    print("Project root:", ROOT)
    print("Model folder:", MODEL_DIR)

    df = load_dataset()
    print("Shape:", df.shape)
    print("Features:", df.shape[1] - 1, "| Rows:", df.shape[0])
    print(df[TARGET_COL].value_counts())
    print(df.head())

    X = df[FEATURE_ORDER].copy()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET_COL].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    test_df = X_test.copy()
    test_df[TARGET_COL] = label_encoder.inverse_transform(y_test)
    test_path = ROOT / "test_data.csv"
    test_df.to_csv(test_path, index=False)

    print("Train size:", len(X_train))
    print("Test size :", len(X_test))
    print("Saved test file:", test_path)
    print("Classes:", list(label_encoder.classes_))

    # --- Logistic Regression ---
    print("=" * 60)
    print("TRAINING: Logistic Regression")
    print("=" * 60)
    lr_pipe = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE),
            ),
        ]
    )
    t0 = time.time()
    lr_pipe.fit(X_train, y_train)
    lr_time = time.time() - t0
    n_iter = lr_pipe.named_steps["classifier"].n_iter_
    print("max_iter setting : 2000")
    print(f"solver iterations used (n_iter_): {n_iter}")
    print(f"training time    : {lr_time:.3f}s")
    print(f"trained on       : {len(X_train)} rows")
    joblib.dump(lr_pipe, MODEL_DIR / "logistic_regression.joblib")
    print("Saved: logistic_regression.joblib")

    # --- Decision Tree ---
    print("=" * 60)
    print("TRAINING: Decision Tree")
    print("=" * 60)
    dt_pipe = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)),
        ]
    )
    t0 = time.time()
    dt_pipe.fit(X_train, y_train)
    dt_time = time.time() - t0
    tree = dt_pipe.named_steps["classifier"]
    print("max_depth setting : 10")
    print(f"actual depth      : {tree.get_depth()}")
    print(f"leaf nodes        : {tree.get_n_leaves()}")
    print(f"training time     : {dt_time:.3f}s")
    joblib.dump(dt_pipe, MODEL_DIR / "decision_tree.joblib")
    print("Saved: decision_tree.joblib")

    # --- kNN ---
    print("=" * 60)
    print("TRAINING: kNN")
    print("=" * 60)
    knn_pipe = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", KNeighborsClassifier(n_neighbors=7)),
        ]
    )
    t0 = time.time()
    knn_pipe.fit(X_train, y_train)
    knn_time = time.time() - t0
    print("n_neighbors : 7")
    print(f"training time: {knn_time:.3f}s")
    print("Note: kNN fitting mainly indexes the training set; distance search happens at predict().")
    joblib.dump(knn_pipe, MODEL_DIR / "knn.joblib")
    print("Saved: knn.joblib")

    # --- Naive Bayes ---
    print("=" * 60)
    print("TRAINING: Naive Bayes (Gaussian)")
    print("=" * 60)
    nb_pipe = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", GaussianNB()),
        ]
    )
    t0 = time.time()
    nb_pipe.fit(X_train, y_train)
    nb_time = time.time() - t0
    nb = nb_pipe.named_steps["classifier"]
    print(f"class priors : {np.round(nb.class_prior_, 4)}")
    print(f"training time: {nb_time:.3f}s")
    joblib.dump(nb_pipe, MODEL_DIR / "naive_bayes.joblib")
    print("Saved: naive_bayes.joblib")

    # --- Random Forest (ensemble rounds) ---
    print("=" * 60)
    print("TRAINING: Random Forest (Ensemble) — tree rounds")
    print("=" * 60)
    tree_rounds = [20, 50, 100, 150, 200]
    rf_history = []
    rf_pre = build_preprocessor()
    X_train_t = rf_pre.fit_transform(X_train)
    X_test_t = rf_pre.transform(X_test)
    rf_clf = RandomForestClassifier(
        n_estimators=0,
        max_depth=12,
        warm_start=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    t0 = time.time()
    for n_trees in tree_rounds:
        rf_clf.set_params(n_estimators=n_trees)
        rf_clf.fit(X_train_t, y_train)
        pred = rf_clf.predict(X_test_t)
        proba = rf_clf.predict_proba(X_test_t)[:, 1]
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, zero_division=0)
        auc = roc_auc_score(y_test, proba)
        rf_history.append(
            {"n_estimators": n_trees, "Accuracy": acc, "F1": f1, "AUC": auc}
        )
        print(
            f"Round n_estimators={n_trees:3d} | "
            f"test Acc={acc:.4f} | test F1={f1:.4f} | test AUC={auc:.4f}"
        )
    rf_time = time.time() - t0
    print(f"Total RF training time: {rf_time:.3f}s")

    rf_pipe = Pipeline(
        [
            ("preprocessor", rf_pre),
            ("classifier", rf_clf),
        ]
    )
    joblib.dump(rf_pipe, MODEL_DIR / "random_forest_ensemble.joblib")
    print("Saved: random_forest_ensemble.joblib")

    hist_df = pd.DataFrame(rf_history)
    print(hist_df)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist_df["n_estimators"], hist_df["Accuracy"], marker="o", label="Test Accuracy")
    ax.plot(hist_df["n_estimators"], hist_df["F1"], marker="o", label="Test F1")
    ax.plot(hist_df["n_estimators"], hist_df["AUC"], marker="o", label="Test AUC")
    ax.set_xlabel("n_estimators (trees added across training rounds)")
    ax.set_ylabel("Score on held-out TEST set")
    ax.set_title("Random Forest training progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    rf_plot = MODEL_DIR / "rf_training_progress.png"
    fig.savefig(rf_plot, dpi=120)
    plt.close(fig)
    print("Saved plot:", rf_plot)

    # --- Evaluate ---
    fitted = {
        "Logistic Regression": lr_pipe,
        "Decision Tree": dt_pipe,
        "kNN": knn_pipe,
        "Naive Bayes": nb_pipe,
        "Random Forest (Ensemble)": rf_pipe,
    }

    metrics_table = {}
    reports = {}
    for name, pipe in fitted.items():
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics_table[name] = evaluate(y_test, y_pred, y_proba)
        reports[name] = {
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test,
                y_pred,
                target_names=[str(c) for c in label_encoder.classes_],
                output_dict=True,
                zero_division=0,
            ),
        }
        print(name, metrics_table[name])

    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.joblib")

    meta = {
        "dataset": "UCI Online Shoppers Purchasing Intention",
        "dataset_url": DATASET_URL,
        "target": TARGET_COL,
        "classes": [str(c) for c in label_encoder.classes_],
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_order": FEATURE_ORDER,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "metrics": metrics_table,
        "reports": reports,
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(meta, f, indent=2)

    metrics_df = pd.DataFrame(metrics_table).T
    print(metrics_df)
    print("Best by F1 :", metrics_df["F1"].idxmax())
    print("Best by AUC:", metrics_df["AUC"].idxmax())

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    for i, (name, pipe) in enumerate(fitted.items()):
        ConfusionMatrixDisplay.from_estimator(
            pipe,
            X_test,
            y_test,
            display_labels=label_encoder.classes_,
            cmap="Blues",
            ax=axes[i],
            colorbar=False,
        )
        axes[i].set_title(name + " (TEST)")
    axes[-1].axis("off")
    plt.tight_layout()
    cm_plot = MODEL_DIR / "confusion_matrices.png"
    fig.savefig(cm_plot, dpi=120)
    plt.close(fig)
    print("Saved plot:", cm_plot)

    notes = {
        "Logistic Regression": "Good accuracy/AUC; recall is low — many purchases missed.",
        "Decision Tree": "Better recall than logistic regression; AUC slightly lower.",
        "kNN": "Decent accuracy; weaker AUC/F1 than tree-based models.",
        "Naive Bayes": "Highest recall, weakest precision/accuracy.",
        "Random Forest (Ensemble)": "Best Accuracy, AUC, F1 and MCC on this dataset.",
    }
    print(pd.DataFrame({"Observation": notes}))
    print("Overall winner:", metrics_df["F1"].idxmax())
    print()
    print("Training complete.")
    print("Next: from project root run  streamlit run app.py")


if __name__ == "__main__":
    main()
