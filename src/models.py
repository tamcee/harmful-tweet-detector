"""
Neural Network Models Module
=============================
Two recurrent architectures for harmful sentiment detection:

1. LSTMModel  — Embedding → LSTM → Dropout → Dense → Output
2. GRUModel   — Embedding → GRU  → Dropout → Dense → Output

Both models use:
- Trainable word embeddings (dim=128)
- Bidirectional recurrent layers (64–128 units)
- Dropout regularization (0.2–0.5)
- Sigmoid output for binary classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    """
    LSTM-based recurrent neural network for sequential text understanding.

    Architecture:
        Embedding(vocab_size, embed_dim)
        → LSTM(embed_dim, hidden_dim, num_layers=2, bidirectional)
        → Dropout
        → Linear(hidden_dim * 2, 64)  (×2 for bidirectional)
        → ReLU + Dropout
        → Linear(64, 1)
        → Sigmoid
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc1 = nn.Linear(hidden_dim * 2, 64)  # *2 for bidirectional
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)                  # (batch, seq_len, embed_dim)
        lstm_out, (hidden, cell) = self.lstm(x) # lstm_out: (batch, seq_len, hidden*2)

        # Use the last hidden state from both directions
        # hidden shape: (num_layers * 2, batch, hidden_dim)
        hidden_fwd = hidden[-2]  # Last layer forward
        hidden_bwd = hidden[-1]  # Last layer backward
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)  # (batch, hidden*2)

        x = self.dropout(hidden_cat)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(1)


class GRUModel(nn.Module):
    """
    GRU-based recurrent neural network for sequential text understanding.

    Architecture:
        Embedding(vocab_size, embed_dim)
        → GRU(embed_dim, hidden_dim, num_layers=2, bidirectional)
        → Dropout
        → Linear(hidden_dim * 2, 64)  (×2 for bidirectional)
        → ReLU + Dropout
        → Linear(64, 1)
        → Sigmoid

    GRU uses fewer parameters than LSTM (no cell state) while maintaining
    comparable performance on many sequence tasks.
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc1 = nn.Linear(hidden_dim * 2, 64)  # *2 for bidirectional
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)                # (batch, seq_len, embed_dim)
        gru_out, hidden = self.gru(x)        # gru_out: (batch, seq_len, hidden*2)

        # Use the last hidden state from both directions
        # hidden shape: (num_layers * 2, batch, hidden_dim)
        hidden_fwd = hidden[-2]  # Last layer forward
        hidden_bwd = hidden[-1]  # Last layer backward
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)  # (batch, hidden*2)

        x = self.dropout(hidden_cat)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(1)


# ─── Model Factory ──────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "lstm": LSTMModel,
    "gru": GRUModel,
}


def get_model(model_name, vocab_size, **kwargs):
    """
    Get a model by name.

    Args:
        model_name: One of 'lstm', 'gru'
        vocab_size: Size of the vocabulary
        **kwargs: Additional model arguments

    Returns:
        nn.Module instance
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY.keys())}")

    model = MODEL_REGISTRY[model_name](vocab_size, **kwargs)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🏗️  Model: {model_name.upper()}")
    print(f"    Total parameters:     {total_params:,}")
    print(f"    Trainable parameters: {trainable:,}")

    return model
