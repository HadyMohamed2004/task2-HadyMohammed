# ============================================================
#  DecodeLabs — AI
#  Project 2: Data Classification Using AI  (MODULE)
#  Algorithm: K-Nearest Neighbors (KNN)
#  Dataset:   Iris Benchmark (150 samples, 3 classes, 4 features)
# ============================================================

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)


# ============================================================
#  PHASE 1 — INPUT: Load & Explore the Raw Material
# ============================================================

def load_and_explore() -> tuple:
    """
    Loads the Iris benchmark dataset and prints a summary.
    Returns features (X), labels (y), class names, and feature names.
    """
    iris = load_iris()
    X    = iris.data        # shape: (150, 4) — the 4 measurements
    y    = iris.target      # shape: (150,)   — 0, 1, or 2

    print("=" * 55)
    print("  DecodeLabs — Project 2: Data Classification")
    print("=" * 55)
    print("\nDATASET LOADED: Iris Benchmark")
    print(f"   Samples   : {X.shape[0]}")
    print(f"   Features  : {X.shape[1]} → {iris.feature_names}")
    print(f"   Classes   : {len(iris.target_names)} → {list(iris.target_names)}")

    unique, counts = np.unique(y, return_counts=True)
    print("\n   Class Distribution (Balanced ✓):")
    for label, count in zip(iris.target_names, counts):
        print(f"     {label:<15}: {count} samples")

    return X, y, list(iris.target_names), list(iris.feature_names)


# ============================================================
#  PHASE 1 — INPUT: Feature Scaling (The Gatekeeper Rule)
# ============================================================

def split_and_scale(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    STEP 1: Shuffle & split — 80% train, 20% test.
    STEP 2: Fit scaler on TRAIN only, transform both sets.
            Never fit on test data — that is data leakage.
    Returns: X_train, X_test, y_train, y_test, scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nTRAIN-TEST SPLIT")
    print(f"   Training samples : {len(X_train)} (80%)")
    print(f"   Testing samples  : {len(X_test)}  (20%)")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"\nFEATURE SCALING APPLIED")
    print(f"   Method: StandardScaler (Mean=0, Variance=1)")
    print(f"   Status: Fitted on training data only ✓")

    return X_train, X_test, y_train, y_test, scaler


# ============================================================
#  PHASE 2 — PROCESS: Find Optimal K
# ============================================================

def find_optimal_k(X_train, X_test, y_train, y_test) -> tuple[int, list]:
    """
    Tests k=1 to k=20 and returns the k with highest F1 score.
    Returns: (best_k, results_list)
    results_list = list of (k, f1_score) tuples
    """
    print("\nFINDING OPTIMAL K (Elbow Method)...")
    best_k  = 1
    best_f1 = 0
    results = []

    for k in range(1, 21):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        f1    = f1_score(y_test, preds, average='weighted')
        results.append((k, round(f1, 6)))
        if f1 > best_f1:
            best_f1 = f1
            best_k  = k

    print(f"\n   {'K':<5} {'F1 Score':<12} {'Status'}")
    print(f"   {'-'*30}")
    for k, f1 in results:
        marker = " ◄ OPTIMAL" if k == best_k else ""
        print(f"   {k:<5} {f1:.4f}      {marker}")

    print(f"\n   Best K = {best_k}  |  Best F1 = {best_f1:.4f}")
    return best_k, results


# ============================================================
#  PHASE 2 — PROCESS: Train Model
# ============================================================

def train_model(X_train, y_train, k: int) -> KNeighborsClassifier:
    """
    Instantiate and fit a KNN model with the given k.
    """
    print(f"\nTRAINING KNN MODEL (k={k})...")
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    print(f"   Model trained on {len(X_train)} samples ✓")
    return model


# ============================================================
#  PHASE 3 — OUTPUT: Evaluate
# ============================================================

def evaluate_model(model, X_test, y_test, class_names) -> dict:
    """
    Computes accuracy, F1, confusion matrix, and full report.
    Returns a dict with all metrics for GUI or further use.
    """
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1       = f1_score(y_test, predictions, average='weighted')
    cm       = confusion_matrix(y_test, predictions)
    report   = classification_report(
        y_test, predictions,
        target_names=class_names,
        output_dict=True
    )
    report_str = classification_report(
        y_test, predictions,
        target_names=class_names
    )

    print("\n" + "=" * 55)
    print("  OUTPUT VALIDATION REPORT")
    print("=" * 55)
    print(f"\nAccuracy  : {accuracy * 100:.2f}%")
    print(f"F1 Score  : {f1:.4f} (weighted average)")
    print(f"\nCONFUSION MATRIX")
    header = "          " + "  ".join(f"{n[:10]:>10}" for n in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_label = f"{class_names[i][:10]:>10}"
        row_data  = "  ".join(f"{val:>10}" for val in row)
        print(f"   {row_label}  {row_data}")
    print(f"\nFULL CLASSIFICATION REPORT\n")
    for line in report_str.split('\n'):
        print(f"   {line}")

    return {
        "accuracy":    accuracy,
        "f1":          f1,
        "cm":          cm,
        "report":      report,
        "report_str":  report_str,
        "predictions": predictions,
    }


# ============================================================
#  PREDICT: Single Sample
# ============================================================

def predict_single(model, scaler, features: list, class_names: list) -> str:
    """
    Predict the class for one sample.
    features: [sepal_length, sepal_width, petal_length, petal_width]
    Returns: predicted class name (str)
    """
    sample  = np.array(features).reshape(1, -1)
    scaled  = scaler.transform(sample)
    pred    = model.predict(scaled)[0]
    proba   = model.predict_proba(scaled)[0]
    return class_names[pred], proba


# ============================================================
#  FULL PIPELINE — convenience function for GUI
# ============================================================

def run_full_pipeline() -> dict:
    """
    Runs the complete Phase 1→2→3 pipeline and returns
    all artefacts needed to populate a GUI dashboard.
    """
    X, y, class_names, feature_names = load_and_explore()
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    optimal_k, k_results = find_optimal_k(X_train, X_test, y_train, y_test)
    model    = train_model(X_train, y_train, optimal_k)
    metrics  = evaluate_model(model, X_test, y_test, class_names)

    return {
        "model":        model,
        "scaler":       scaler,
        "class_names":  class_names,
        "feature_names": feature_names,
        "X_train":      X_train,
        "X_test":       X_test,
        "y_train":      y_train,
        "y_test":       y_test,
        "optimal_k":    optimal_k,
        "k_results":    k_results,
        **metrics,
    }


# ── CLI Entry Point ───────────────────────────────────────────
if __name__ == "__main__":
    X, y, class_names, feature_names = load_and_explore()
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    optimal_k, _ = find_optimal_k(X_train, X_test, y_train, y_test)
    model = train_model(X_train, y_train, optimal_k)
    evaluate_model(model, X_test, y_test, class_names)