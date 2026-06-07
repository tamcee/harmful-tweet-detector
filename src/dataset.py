"""
PyTorch Dataset Module
======================
Creates PyTorch Dataset and DataLoader objects for training, validation, and testing.
Handles train/val/test splitting.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class TweetDataset(Dataset):
    """
    PyTorch Dataset for tweet sentiment classification.

    Args:
        sequences: List of padded integer sequences
        labels: List of integer labels (0 or 1)
    """

    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def create_dataloaders(sequences, labels, batch_size=64, test_size=0.2, val_size=0.1, seed=42):
    """
    Split data and create DataLoaders for train, validation, and test sets.

    Split ratios:
        - Train: 70%
        - Validation: 10%
        - Test: 20%

    Args:
        sequences: List of padded integer sequences
        labels: List of labels
        batch_size: Batch size for DataLoaders
        test_size: Fraction for test split
        val_size: Fraction for validation split (from remaining after test)
        seed: Random seed

    Returns:
        train_loader, val_loader, test_loader
    """
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        sequences, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    # Second split: train vs val
    val_fraction = val_size / (1 - test_size)  # Adjust fraction for trainval set
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction, random_state=seed, stratify=y_trainval
    )

    # Create datasets
    train_dataset = TweetDataset(X_train, y_train)
    val_dataset = TweetDataset(X_val, y_val)
    test_dataset = TweetDataset(X_test, y_test)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"📊 Data split:")
    print(f"   Train:      {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"   Validation: {len(val_dataset)} samples ({len(val_loader)} batches)")
    print(f"   Test:       {len(test_dataset)} samples ({len(test_loader)} batches)")

    return train_loader, val_loader, test_loader
