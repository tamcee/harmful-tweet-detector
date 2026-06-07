"""
Training Module
===============
Implements the training loop with:
- Backpropagation for weight updates
- Configurable optimizers (SGD, RMSprop, Adam)
- Learning rate scheduling
- Training and validation loss/accuracy tracking
- Early stopping
"""

import time
import torch
import torch.nn as nn


# ─── Optimizer Factory ──────────────────────────────────────────────────────────

def get_optimizer(model, optimizer_name="adam", lr=0.001, weight_decay=0.0):
    """
    Get an optimizer by name.

    Args:
        model: nn.Module
        optimizer_name: One of 'sgd', 'rmsprop', 'adam'
        lr: Learning rate
        weight_decay: L2 regularization

    Returns:
        torch.optim.Optimizer
    """
    optimizers = {
        "sgd": torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay),
        "rmsprop": torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay),
        "adam": torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay),
    }

    if optimizer_name.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Choose from {list(optimizers.keys())}")

    return optimizers[optimizer_name.lower()]


# ─── Training Loop ──────────────────────────────────────────────────────────────

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer_name="adam",
    lr=0.001,
    weight_decay=0.0,
    epochs=20,
    device="cpu",
    patience=5,
    verbose=True,
):
    """
    Train a binary classification model.

    Uses:
    - Binary Cross Entropy loss
    - Backpropagation for gradient computation
    - Configurable optimizer for weight updates
    - Early stopping based on validation loss

    Args:
        model: nn.Module
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        optimizer_name: 'sgd', 'rmsprop', or 'adam'
        lr: Learning rate
        epochs: Maximum number of epochs
        device: 'cpu' or 'cuda'
        patience: Early stopping patience
        verbose: Print progress

    Returns:
        history: dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc' lists
    """
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = get_optimizer(model, optimizer_name, lr, weight_decay)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    if verbose:
        print(f"\n🚀 Training with optimizer: {optimizer_name.upper()}")
        print(f"   Learning rate: {lr}, Epochs: {epochs}, Device: {device}")
        print(f"   {'─' * 60}")

    start_time = time.time()

    for epoch in range(epochs):
        # ── Training phase ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_x)
            
            # Apply Label Smoothing to prevent overconfidence
            batch_y_smooth = batch_y * 0.8 + 0.1
            loss = criterion(outputs, batch_y_smooth)

            # Backward pass (backpropagation)
            optimizer.zero_grad()
            loss.backward()

            # Weight update (gradient descent step)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            predictions = (outputs >= 0.5).float()
            train_correct += (predictions == batch_y).sum().item()
            train_total += batch_y.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ── Validation phase ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                predictions = (outputs >= 0.5).float()
                val_correct += (predictions == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if verbose:
            print(
                f"   Epoch {epoch+1:3d}/{epochs} │ "
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} │ "
                f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"   ⏹  Early stopping at epoch {epoch+1} (patience={patience})")
                break

    elapsed = time.time() - start_time

    # Load best model weights
    if best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(f"   ✅ Training complete in {elapsed:.1f}s | Best val loss: {best_val_loss:.4f}")

    return history
