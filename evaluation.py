"""Step 4: evaluate the models and explain the results. Charts are saved in outputs/figures."""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, classification_report,
                             f1_score, precision_score, recall_score, roc_auc_score)

from config import FIG_DIR


def _save(name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=150)
    plt.close()
    print("  saved:", FIG_DIR.name + "/" + name)


def metrics_row(name, model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {"Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, proba)}


def evaluate_models(models, X_test, y_test):
    rows = [metrics_row(name, m, X_test, y_test) for name, m in models.items()]
    return pd.DataFrame(rows).set_index("Model").round(3).sort_values("ROC-AUC", ascending=False)


def plot_confusion_and_roc(models, best_name, X_test, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_estimator(models[best_name], X_test, y_test, cmap="Blues", ax=axes[0])
    axes[0].set_title(f"Confusion matrix: {best_name}")
    for name, m in models.items():
        RocCurveDisplay.from_estimator(m, X_test, y_test, name=name, ax=axes[1])
    axes[1].plot([0, 1], [0, 1], "k--", label="Random guess")
    axes[1].set_title("ROC curves")
    axes[1].legend()
    _save("model_confusion_matrix_and_roc.png")


def print_report(model, X_test, y_test):
    print(classification_report(y_test, model.predict(X_test)))


def plot_feature_importance(tuned_rf):
    names = tuned_rf.named_steps["pre"].get_feature_names_out()
    importance = pd.Series(tuned_rf.named_steps["clf"].feature_importances_, index=names)
    top = importance.sort_values(ascending=False).head(10)
    top.sort_values().plot(kind="barh", figsize=(7, 4.5), color="#2E6F9E")
    plt.title("Top 10 most important features")
    plt.xlabel("Importance")
    _save("model_feature_importance.png")
    return top
