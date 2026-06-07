"""
Context-Aware Harmful Sentiment Detection in Twitter Threads
=============================================================
Main orchestration script that:
1. Prepares the dataset
2. Preprocesses text data
3. Validates using 5-Fold Cross Validation
4. Trains LSTM and GRU models dynamically
5. Evaluates and generates visualization plots
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.prepare_data import prepare_and_save
from src.preprocessing import preprocess_texts, clean_text, tokenize, Vocabulary, pad_sequence
from src.dataset import create_dataloaders
from src.models import get_model
from src.train import train_model
from src.evaluate import (
    evaluate_model,
    print_evaluation,
    plot_training_curves,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_metrics_comparison,
    save_results_table,
)


# ─── Configuration ──────────────────────────────────────────────────────────────

CONFIG = {
    "n_samples": 5000,  # Focus on a balanced subset of the real dataset
    "max_vocab_size": 10000,
    "max_seq_len": 100,
    "embed_dim": 64,  # Reduced complexity
    "hidden_dim": 64, # Explicitly lower hidden dimension
    "batch_size": 64,
    "epochs": 20,
    "learning_rate": 0.001,
    "weight_decay": 1e-4, # L2 Regularization
    "dropout": 0.5,   # Increased dropout for better generalization
    "patience": 5,
    "seed": 42,
    "results_dir": "results",
}

MODELS = ["lstm", "gru"]
OPTIMIZERS = ["adam"]



def main():
    """Main execution pipeline."""
    total_start = time.time()

    print("=" * 80)
    print("  CONTEXT-AWARE HARMFUL SENTIMENT DETECTION IN TWITTER THREADS")
    print("  Deep Learning Lab (AIM 3230) — Mini Project")
    print("  Architecture: LSTM vs GRU Comparison")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n⚙️  Device: {device}")
    print(f"⚙️  Configuration: {CONFIG}")

    # ── Step 1: Data Preparation ──
    print("\n" + "=" * 80)
    print("📦 STEP 1: DATA PREPARATION")
    print("=" * 80)

    df = prepare_and_save(n_samples=CONFIG["n_samples"])

    # ── Step 2: Text Preprocessing ──
    print("\n" + "=" * 80)
    print("🔧 STEP 2: TEXT PREPROCESSING")
    print("=" * 80)

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    padded_sequences, vocab = preprocess_texts(
        texts,
        max_vocab_size=CONFIG["max_vocab_size"],
        max_seq_len=CONFIG["max_seq_len"],
    )

    # ── Step 3: Create DataLoaders ──
    print("\n" + "=" * 80)
    print("📊 STEP 3: CREATING DATA LOADERS")
    print("=" * 80)

    train_loader, val_loader, test_loader = create_dataloaders(
        padded_sequences,
        labels,
        batch_size=CONFIG["batch_size"],
        seed=CONFIG["seed"],
    )

    print("\n" + "=" * 80)
    print("🔄 STEP 3.5: 5-FOLD CROSS VALIDATION")
    print("=" * 80)

    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from src.dataset import TweetDataset
    from torch.utils.data import DataLoader

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CONFIG["seed"])
    sequences_arr = np.array(padded_sequences)
    labels_arr = np.array(labels)

    for model_name in MODELS:
        print(f"\nRunning 5-Fold CV for {model_name.upper()}...")
        fold_f1s = []
        # Suppress excessive prints for CV
        for fold, (train_idx, val_idx) in enumerate(skf.split(sequences_arr, labels_arr)):
            X_train, X_val = sequences_arr[train_idx], sequences_arr[val_idx]
            y_train, y_val = labels_arr[train_idx], labels_arr[val_idx]
            
            train_loader_cv = DataLoader(TweetDataset(X_train, y_train), batch_size=CONFIG["batch_size"], shuffle=True)
            val_loader_cv = DataLoader(TweetDataset(X_val, y_val), batch_size=CONFIG["batch_size"], shuffle=False)
            
            model_cv = get_model(
                model_name,
                vocab_size=len(vocab),
                embed_dim=CONFIG["embed_dim"],
                dropout=CONFIG["dropout"],
            )
            
            # Fast training for CV
            train_model(
                model_cv,
                train_loader_cv,
                val_loader_cv,
                optimizer_name="adam",
                lr=CONFIG["learning_rate"],
                weight_decay=CONFIG["weight_decay"],
                epochs=10,
                device=device,
                patience=3,
                verbose=False
            )
            
            metrics_cv, _, _ = evaluate_model(model_cv, val_loader_cv, device)
            fold_f1s.append(metrics_cv["f1"])
            print(f"  Fold {fold+1} F1-Score: {metrics_cv['f1']:.4f}")
            
        print(f"🏆 {model_name.upper()} Average CV F1-Score: {np.mean(fold_f1s):.4f} (±{np.std(fold_f1s):.4f})")

    # ── Step 4: Final Training & Evaluation (LSTM vs GRU) ──
    print("\n" + "=" * 80)
    print("🚀 STEP 4: FINAL TRAINING & EVALUATION")
    print("=" * 80)

    all_results = {}
    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    for model_name in MODELS:
        for opt_name in OPTIMIZERS:
            key = f"{model_name}_{opt_name}"
            print(f"\n{'═' * 70}")
            print(f"  Experiment: {model_name.upper()} + {opt_name.upper()}")
            print(f"{'═' * 70}")

            # Create model
            model = get_model(
                model_name,
                vocab_size=len(vocab),
                embed_dim=CONFIG["embed_dim"],
                dropout=CONFIG["dropout"],
            )

            # Train
            history = train_model(
                model,
                train_loader,
                val_loader,
                optimizer_name=opt_name,
                lr=CONFIG["learning_rate"],
                weight_decay=CONFIG["weight_decay"],
                epochs=CONFIG["epochs"],
                device=device,
                patience=CONFIG["patience"],
            )

            # Evaluate
            metrics, preds, labels_arr = evaluate_model(model, test_loader, device)
            print_evaluation(metrics, model_name, opt_name)

            # Save plots
            plot_training_curves(history, model_name, opt_name, CONFIG["results_dir"])
            plot_confusion_matrix(labels_arr, preds, model_name, opt_name, CONFIG["results_dir"])

            # Store results
            all_results[key] = {
                "metrics": metrics,
                "history": history,
                "model": model,
            }

    # ── Step 5: Comparison Plots ──
    print("\n" + "=" * 80)
    print("📊 STEP 5: GENERATING COMPARISON PLOTS")
    print("=" * 80)

    best_results = plot_model_comparison(all_results, CONFIG["results_dir"])
    plot_metrics_comparison(all_results, CONFIG["results_dir"])
    results_text = save_results_table(all_results, CONFIG["results_dir"])

    # ── Step 6: Export Best Model ──
    best_key = max(all_results.keys(), key=lambda k: all_results[k]["metrics"]["f1"])
    best_model = all_results[best_key]["model"]
    print(f"\n🏆 Saving best verified model: {best_key.replace('_', ' + ').upper()}")
    
    # Save the best model and vocab for app.py
    model_path = os.path.join(CONFIG["results_dir"], "best_model.pth")
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'model_name': best_key.split('_')[0],
        'config': CONFIG
    }, model_path)
    print(f"💾 Saved best model weights → {model_path}")
    
    vocab_path = os.path.join(CONFIG["results_dir"], "vocab.json")
    vocab.save(vocab_path)
    print(f"💾 Saved vocabulary → {vocab_path}")

    # ── Summary ──
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print("✅ ALL EXPERIMENTS COMPLETE")
    print(f"   Total time: {total_elapsed:.1f}s")
    print(f"   Results saved to: {CONFIG['results_dir']}/")
    print("=" * 80)

    # List all generated files
    print("\n📁 Generated files:")
    for f in sorted(os.listdir(CONFIG["results_dir"])):
        filepath = os.path.join(CONFIG["results_dir"], f)
        size = os.path.getsize(filepath)
        print(f"   {f} ({size:,} bytes)")


if __name__ == "__main__":
    main()
