"""
visualization.py
----------------
Funções de visualização para EDA e análise de resultados.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Estilo global
plt.rcParams.update({
    "figure.dpi": 120,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FRAUD_COLOR = "#E74C3C"
LEGIT_COLOR = "#2ECC71"
PALETTE = [LEGIT_COLOR, FRAUD_COLOR]


def plot_class_distribution(y: pd.Series):
    """Plota a distribuição de classes com contagem e percentual."""
    counts = y.value_counts()
    pcts = y.value_counts(normalize=True) * 100
    labels = ["Legítima", "Fraude"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Barras
    bars = axes[0].bar(labels, counts.values, color=PALETTE, edgecolor="white", width=0.5)
    for bar, count, pct in zip(bars, counts.values, pcts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                     f"{count:,}\n({pct:.2f}%)", ha="center", va="bottom", fontsize=10)
    axes[0].set_title("Distribuição de Classes (Absoluto)", fontsize=12)
    axes[0].set_ylabel("Número de transações")
    axes[0].set_yscale("log")
    axes[0].grid(axis="y", alpha=0.3)

    # Pizza
    axes[1].pie(counts.values, labels=labels, colors=PALETTE, autopct="%1.3f%%",
                startangle=140, wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Proporção de Classes", fontsize=12)

    plt.suptitle("Dataset Altamente Desbalanceado", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_amount_distribution(df: pd.DataFrame):
    """Compara a distribuição do valor das transações por classe."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, col, title in zip(
        axes,
        ["Amount", "log_amount"],
        ["Valor Original (Amount)", "Valor Log-transformado (log1p)"],
    ):
        for cls, color, label in zip([0, 1], PALETTE, ["Legítima", "Fraude"]):
            subset = df[df["Class"] == cls][col]
            ax.hist(subset, bins=60, alpha=0.6, color=color, label=label, density=True)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel("Densidade")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Distribuição do Valor das Transações por Classe", fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


def plot_time_analysis(df: pd.DataFrame):
    """Analisa padrões temporais de fraude ao longo do tempo."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)

    # Transações ao longo do tempo
    bins = np.linspace(0, df["Time"].max(), 100)
    for cls, color, label in zip([0, 1], PALETTE, ["Legítima", "Fraude"]):
        subset = df[df["Class"] == cls]["Time"]
        axes[0].hist(subset, bins=bins, alpha=0.6, color=color, label=label, density=True)
    axes[0].set_title("Distribuição Temporal das Transações (densidade)")
    axes[0].set_xlabel("Tempo (segundos desde primeira transação)")
    axes[0].set_ylabel("Densidade")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Fraudes por hora do dia
    if "hour_of_day" in df.columns:
        fraud_by_hour = df.groupby(["hour_of_day", "Class"]).size().unstack(fill_value=0)
        fraud_rate_by_hour = (fraud_by_hour[1] / (fraud_by_hour[0] + fraud_by_hour[1])) * 100
        axes[1].plot(fraud_rate_by_hour.index, fraud_rate_by_hour.values,
                     color=FRAUD_COLOR, lw=2, marker="o", markersize=3)
        axes[1].fill_between(fraud_rate_by_hour.index, fraud_rate_by_hour.values, alpha=0.2, color=FRAUD_COLOR)
        axes[1].set_title("Taxa de Fraude por Hora do Dia (%)")
        axes[1].set_xlabel("Hora do Dia (estimada)")
        axes[1].set_ylabel("% de transações fraudulentas")
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_feature_correlation(df: pd.DataFrame, n_features: int = 20):
    """Plota correlação das features com o target."""
    corr = df.corr()["Class"].drop("Class").abs().sort_values(ascending=False).head(n_features)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [FRAUD_COLOR if v > 0.1 else "#5D6D7E" for v in corr.values]
    bars = ax.barh(corr.index[::-1], corr.values[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Correlação Absoluta com Classe Fraude")
    ax.set_title(f"Top {n_features} Features – Correlação com Target", fontsize=12)
    ax.axvline(0.1, color="red", linestyle="--", lw=1, alpha=0.6, label="Threshold 0.10")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_feature_importance(importances: np.ndarray, feature_names: list, model_name: str = "Model", top_n: int = 20):
    """Plota importância das features de um modelo tree-based."""
    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(
        range(len(indices)),
        importances[indices],
        color="#3498DB",
        edgecolor="white",
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importância")
    ax.set_title(f"Feature Importance – {model_name} (Top {top_n})", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig
