"""Plotting helpers."""
from pathlib import Path
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, output_path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["Predicted ↓", "Predicted ↑"])
    ax.set_yticks([0, 1], labels=["Actual ↓", "Actual ↑"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set(title="Confusion Matrix", xlabel="Predicted Class", ylabel="Actual Class")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_curve(fpr, tpr, auc, output_path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"Model (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax.set(title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
