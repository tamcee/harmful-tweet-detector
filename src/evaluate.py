"""
Evaluation Module
=================
Provides functions for:
- Model evaluation (accuracy, precision, recall, F1-score)
- Confusion matrix generation
- Visualization of training curves and model comparisons
- Side-by-side LSTM vs GRU metrics comparison
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ─── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(model, test_loader, device="cpu"):
    """
    Evaluate a trained model on the test set.

    Returns:
        metrics: dict with accuracy, precision, recall, f1
        all_preds: numpy array of predictions
        all_labels: numpy array of true labels
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            predictions = (outputs >= 0.5).float().cpu().numpy()
            all_preds.extend(predictions)
            all_labels.extend(batch_y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }

    return metrics, all_preds, all_labels


def print_evaluation(metrics, model_name, optimizer_name):
    """Print formatted evaluation metrics."""
    print(f"\n📊 Evaluation Results — {model_name.upper()} + {optimizer_name.upper()}")
    print(f"   {'─' * 40}")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-Score:  {metrics['f1']:.4f}")


# ─── Visualization Functions ────────────────────────────────────────────────────

# Set the style globally
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def plot_training_curves(history, model_name, optimizer_name, save_dir="results"):
    """
    Plot training and validation loss/accuracy curves.

    Saves two subplots: loss curves and accuracy curves.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", linewidth=2, markersize=4)
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss", linewidth=2, markersize=4)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title(f"Loss Curves — {model_name.upper()} + {optimizer_name.upper()}", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(epochs, history["train_acc"], "o-", label="Train Accuracy", linewidth=2, markersize=4)
    ax2.plot(epochs, history["val_acc"], "s-", label="Val Accuracy", linewidth=2, markersize=4)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title(f"Accuracy Curves — {model_name.upper()} + {optimizer_name.upper()}", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    filepath = os.path.join(save_dir, f"curves_{model_name}_{optimizer_name}.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   📈 Saved training curves → {filepath}")


def plot_confusion_matrix(all_labels, all_preds, model_name, optimizer_name, save_dir="results"):
    """Plot and save a confusion matrix heatmap."""
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Harmful", "Harmful"],
        yticklabels=["Non-Harmful", "Harmful"],
        ax=ax,
        annot_kws={"size": 16},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title(
        f"Confusion Matrix — {model_name.upper()} + {optimizer_name.upper()}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    filepath = os.path.join(save_dir, f"confusion_{model_name}_{optimizer_name}.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   🔲 Saved confusion matrix → {filepath}")


def plot_model_comparison(all_results, save_dir="results"):
    """
    Create a comprehensive LSTM vs GRU model comparison chart.

    Shows metrics (accuracy, precision, recall, F1) for each model+optimizer,
    plus validation loss convergence.
    """
    os.makedirs(save_dir, exist_ok=True)

    models = ["lstm", "gru"]
    optimizers = ["adam", "rmsprop"]
    metrics_names = ["accuracy", "precision", "recall", "f1"]

    # Find best optimizer for each model (by F1)
    best_results = {}
    for model in models:
        best_f1 = -1
        best_opt = None
        for opt in optimizers:
            key = f"{model}_{opt}"
            if key in all_results:
                f1 = all_results[key]["metrics"]["f1"]
                if f1 > best_f1:
                    best_f1 = f1
                    best_opt = opt
        if best_opt:
            best_results[model] = {
                "optimizer": best_opt,
                "metrics": all_results[f"{model}_{best_opt}"]["metrics"],
            }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ── Left: Grouped bar chart — all 4 experiments ──
    x = np.arange(len(metrics_names))
    width = 0.18
    colors = ["#7b1fa2", "#9c27b0", "#2e7d32", "#4caf50"]
    labels_list = []
    idx = 0
    for model in models:
        for opt in optimizers:
            key = f"{model}_{opt}"
            if key in all_results:
                values = [all_results[key]["metrics"][m] for m in metrics_names]
                label = f"{model.upper()} + {opt.upper()}"
                bars = ax1.bar(
                    x + idx * width, values, width,
                    label=label, color=colors[idx], alpha=0.85,
                )
                for bar, val in zip(bars, values):
                    ax1.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.3f}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold",
                    )
                labels_list.append(label)
                idx += 1

    ax1.set_xlabel("Metric", fontsize=12)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_title("LSTM vs GRU — All Metrics", fontsize=13, fontweight="bold")
    ax1.set_xticks(x + width * (idx - 1) / 2)
    ax1.set_xticklabels([m.capitalize() for m in metrics_names], fontsize=11)
    ax1.set_ylim(0, 1.15)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # ── Right: Validation loss convergence ──
    line_colors = {"lstm": "#7b1fa2", "gru": "#2e7d32"}
    line_styles = {"adam": "-", "rmsprop": "--"}
    for model in models:
        for opt in optimizers:
            key = f"{model}_{opt}"
            if key in all_results and "history" in all_results[key]:
                val_losses = all_results[key]["history"]["val_loss"]
                ax2.plot(
                    range(1, len(val_losses) + 1),
                    val_losses,
                    f"o{line_styles[opt]}",
                    label=f"{model.upper()} + {opt.upper()}",
                    linewidth=2,
                    markersize=4,
                    color=line_colors[model],
                    alpha=0.9 if opt == "adam" else 0.6,
                )

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Validation Loss", fontsize=12)
    ax2.set_title("Validation Loss Convergence", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "LSTM vs GRU — Model Architecture Comparison",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    filepath = os.path.join(save_dir, "model_comparison.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Saved model comparison → {filepath}")

    return best_results


def plot_metrics_comparison(all_results, save_dir="results"):
    """
    Create a focused metrics comparison: LSTM vs GRU side-by-side bar chart
    for accuracy, precision, recall, and F1-score (using best optimizer for each).
    """
    os.makedirs(save_dir, exist_ok=True)

    models = ["lstm", "gru"]
    optimizers = ["adam", "rmsprop"]
    metrics_names = ["accuracy", "precision", "recall", "f1"]
    display_names = ["Accuracy", "Precision", "Recall", "F1-Score"]

    # Find best optimizer for each model
    best = {}
    for model in models:
        best_f1 = -1
        best_opt = None
        for opt in optimizers:
            key = f"{model}_{opt}"
            if key in all_results:
                f1 = all_results[key]["metrics"]["f1"]
                if f1 > best_f1:
                    best_f1 = f1
                    best_opt = opt
        if best_opt:
            best[model] = all_results[f"{model}_{best_opt}"]["metrics"]

    if len(best) < 2:
        print("⚠️  Not enough results for metrics comparison")
        return

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    colors = {"lstm": "#7b1fa2", "gru": "#2e7d32"}

    for i, (metric, display) in enumerate(zip(metrics_names, display_names)):
        ax = axes[i]
        vals = [best[m][metric] for m in models]
        bars = ax.bar(
            [m.upper() for m in models], vals,
            color=[colors[m] for m in models],
            alpha=0.85, width=0.5,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )
        ax.set_title(display, fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylabel("Score" if i == 0 else "", fontsize=12)

    fig.suptitle(
        "LSTM vs GRU — Evaluation Metrics (Best Optimizer)",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    filepath = os.path.join(save_dir, "metrics_comparison.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Saved metrics comparison → {filepath}")


def save_results_table(all_results, save_dir="results"):
    """Save a formatted results table to a text file."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, "results_table.txt")

    lines = []
    lines.append("=" * 80)
    lines.append("CONTEXT-AWARE HARMFUL SENTIMENT DETECTION — RESULTS SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'Model':<15} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    lines.append("─" * 67)

    best_f1 = 0
    best_key = ""

    for key, result in sorted(all_results.items()):
        parts = key.split("_")
        model = parts[0]
        m = result["metrics"]
        lines.append(
            f"{model.upper():<15} "
            f"{m['accuracy']*100:>11.2f}% "
            f"{m['precision']*100:>11.2f}% "
            f"{m['recall']*100:>11.2f}% "
            f"{m['f1']:>12.4f}"
        )
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_key = key

    lines.append("─" * 67)
    parts = best_key.split("_")
    lines.append(f"\n🏆 Best model: {parts[0].upper()} (F1: {best_f1:.4f})")
    lines.append("")

    result_text = "\n".join(lines)
    with open(filepath, "w") as f:
        f.write(result_text)

    print(f"\n📋 Saved results table → {filepath}")
    print(result_text)

    return result_text
