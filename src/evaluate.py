"""
evaluate.py
-----------
Medical-grade evaluation: accuracy alone is misleading for diagnostic
tasks. This module reports per-class precision/recall/F1, sensitivity,
specificity, a confusion matrix, and ROC-AUC — the metrics a clinical
ML reviewer would actually ask for.

In this problem, a false negative (real tumor predicted as "notumor")
is far more costly than a false positive, so recall/sensitivity on the
tumor classes matters more than overall accuracy.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from preprocessing import CLASS_NAMES

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def evaluate_model(model, test_gen, save_prefix: str = "model"):
    test_gen.reset()
    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n=== Classification Report ===")
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print(report)
    with open(REPORTS_DIR / f"{save_prefix}_classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, save_prefix)
    sensitivity_specificity(cm, save_prefix)
    plot_roc_curves(y_true, y_pred_probs, save_prefix)

    return {"y_true": y_true, "y_pred": y_pred, "y_pred_probs": y_pred_probs, "confusion_matrix": cm}


def plot_confusion_matrix(cm, save_prefix: str):
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — Brain Tumor MRI Classification")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"{save_prefix}_confusion_matrix.png", dpi=150)
    plt.close()


def sensitivity_specificity(cm, save_prefix: str):
    """Per-class sensitivity (recall) and specificity via one-vs-rest."""
    n_classes = cm.shape[0]
    lines = []
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sensitivity = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        line = f"{CLASS_NAMES[i]:12s}  sensitivity={sensitivity:.4f}  specificity={specificity:.4f}"
        print(line)
        lines.append(line)
    with open(REPORTS_DIR / f"{save_prefix}_sens_spec.txt", "w") as f:
        f.write("\n".join(lines))


def plot_roc_curves(y_true, y_pred_probs, save_prefix: str):
    y_true_bin = label_binarize(y_true, classes=list(range(len(CLASS_NAMES))))
    plt.figure(figsize=(7, 6))
    for i, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — One-vs-Rest per Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"{save_prefix}_roc_curves.png", dpi=150)
    plt.close()
