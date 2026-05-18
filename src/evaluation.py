import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
)


def evaluate_model(model, X_test, y_test, model_name: str = "Model", threshold: float = 0.5) -> dict:
    """Avalia um modelo e retorna um dicionário com as principais métricas."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    recall_fraud = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision_fraud = tp / (tp + fp) if (tp + fp) > 0 else 0

    metrics = {
        "model": model_name,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "f1_score": round(f1, 4),
        "recall_fraud": round(recall_fraud, 4),
        "precision_fraud": round(precision_fraud, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }

    print(f"\n{'='*50}")
    print(f"{model_name}")
    print(f"{'='*50}")
    print(f"  ROC-AUC:          {roc_auc:.4f}")
    print(f"  PR-AUC:           {pr_auc:.4f}")
    print(f"  F1-Score:         {f1:.4f}")
    print(f"  Recall (fraude):  {recall_fraud:.4f}")
    print(f"  Precision (fraud):{precision_fraud:.4f}")
    print(f"\n  Fraudes capturadas: {tp}/{tp+fn} ({recall_fraud*100:.1f}%)")
    print(f"  Falsos alarmes:     {fp}")

    return metrics


def plot_confusion_matrix(model, X_test, y_test, model_name: str = "Model", threshold: float = 0.5, ax=None):
    """Plota a matriz de confusão normalizada e absoluta."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legítima", "Fraude"],
        yticklabels=["Legítima", "Fraude"],
        ax=ax,
    )
    ax.set_title(f"Matriz de Confusão\n{model_name}", fontsize=11)
    ax.set_ylabel("Real")
    ax.set_xlabel("Previsto")
    return ax


def plot_roc_curves(models_dict: dict, X_test, y_test):
    """Plota curvas ROC para múltiplos modelos."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title("Curvas ROC – Comparação de Modelos")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_precision_recall_curves(models_dict: dict, X_test, y_test):
    """
    Plota curvas Precision-Recall para múltiplos modelos.
    Mais informativa que ROC em datasets desbalanceados.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax.plot(recall, precision, lw=2, label=f"{name} (AP={ap:.3f})")

    ax.axhline(y=y_test.mean(), color="k", linestyle="--", lw=1, label=f"Baseline ({y_test.mean():.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall – Comparação de Modelos")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_threshold_analysis(model, X_test, y_test, model_name: str = "Model"):
    """
    Analisa como F1, Precision e Recall variam com o threshold.
    Útil para escolher o threshold ideal em produção.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 200)
    
    f1s, precisions, recalls = [], [], []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)

    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, f1s, label="F1-Score", lw=2)
    ax.plot(thresholds, precisions, label="Precision", lw=2, linestyle="--")
    ax.plot(thresholds, recalls, label="Recall", lw=2, linestyle=":")
    ax.axvline(best_threshold, color="red", linestyle="-.", lw=1.5,
               label=f"Melhor threshold: {best_threshold:.2f} (F1={f1s[best_idx]:.3f})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Análise de Threshold – {model_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    print(f"Melhor threshold: {best_threshold:.3f} → F1={f1s[best_idx]:.4f}")
    return fig, best_threshold


def build_results_table(results_list: list) -> pd.DataFrame:
    """Monta uma tabela comparativa de todos os modelos."""
    df = pd.DataFrame(results_list).set_index("model")
    df = df[["roc_auc", "pr_auc", "f1_score", "recall_fraud", "precision_fraud"]]
    df.columns = ["ROC-AUC", "PR-AUC", "F1-Score", "Recall (Fraude)", "Precision (Fraude)"]
    return df.sort_values("PR-AUC", ascending=False)
