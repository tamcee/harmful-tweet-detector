"""
Text Preprocessing Module
=========================
Handles text cleaning, tokenization, vocabulary building,
and sequence encoding for Twitter thread data.

Preprocessing steps (as specified in the synopsis):
- Removal of URLs, mentions, hashtags, emojis, and punctuation
- Tokenization and sequence padding
- Conversion of text into numerical format
"""

import re
import string
from collections import Counter

import nltk

# Download required NLTK data (run once)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

# ─── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_text(text):
    """
    Clean a single tweet:
    1. Lowercase
    2. Remove URLs
    3. Remove @mentions
    4. Remove #hashtags (keep the word after #)
    5. Remove emojis and special Unicode
    6. Remove punctuation
    7. Remove extra whitespace
    """
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtag symbol but keep the word
    text = re.sub(r"#(\w+)", r"\1", text)

    # Remove emojis and non-ASCII characters
    text = re.sub(
        r"[^\x00-\x7F]+",  # Remove non-ASCII (emojis, special chars)
        "",
        text,
    )

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ─── Tokenization ──────────────────────────────────────────────────────────────

def tokenize(text):
    """Tokenize text into words using NLTK word_tokenize."""
    tokens = word_tokenize(text)
    # Remove stopwords and very short tokens
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return tokens


# ─── Vocabulary ─────────────────────────────────────────────────────────────────

class Vocabulary:
    """
    Builds a word-to-index vocabulary from a corpus of tokenized texts.

    Special tokens:
        <PAD> : 0  (padding)
        <UNK> : 1  (unknown word)
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        self.word_counts = Counter()

    def build(self, tokenized_texts):
        """
        Build vocabulary from a list of tokenized texts.

        Args:
            tokenized_texts: List of lists of tokens
        """
        for tokens in tokenized_texts:
            self.word_counts.update(tokens)

        # Most common words (minus the 2 special tokens)
        most_common = self.word_counts.most_common(self.max_vocab_size - 2)

        for word, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"📖 Vocabulary built: {len(self.word2idx)} words "
              f"(from {len(self.word_counts)} unique tokens)")

    def encode(self, tokens):
        """Convert a list of tokens to a list of indices."""
        return [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in tokens]

    def decode(self, indices):
        """Convert a list of indices back to tokens."""
        return [self.idx2word.get(i, self.UNK_TOKEN) for i in indices]

    def __len__(self):
        return len(self.word2idx)

    def save(self, filepath):
        import json
        with open(filepath, 'w') as f:
            json.dump({
                'word2idx': self.word2idx,
                'idx2word': {int(k): v for k, v in self.idx2word.items()},
                'max_vocab_size': self.max_vocab_size
            }, f)

    @classmethod
    def load(cls, filepath):
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        vocab = cls(max_vocab_size=data.get('max_vocab_size', 10000))
        vocab.word2idx = data['word2idx']
        vocab.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        return vocab
# ─── Sequence Padding ──────────────────────────────────────────────────────────

def pad_sequence(encoded, max_len=50):
    """
    Pad or truncate an encoded sequence to a fixed length.

    Args:
        encoded: List of integer indices
        max_len: Target length

    Returns:
        List of length max_len
    """
    if len(encoded) >= max_len:
        return encoded[:max_len]
    else:
        return encoded + [0] * (max_len - len(encoded))


# ─── Full Pipeline ──────────────────────────────────────────────────────────────

def preprocess_texts(texts, vocab=None, max_vocab_size=10000, max_seq_len=50, fit=True):
    """
    Full preprocessing pipeline:
    1. Clean each text
    2. Tokenize
    3. Build vocabulary (if fit=True)
    4. Encode to indices
    5. Pad sequences

    Args:
        texts: List of raw tweet strings
        vocab: Existing Vocabulary (optional)
        max_vocab_size: Max vocabulary size (if building new)
        max_seq_len: Max sequence length for padding
        fit: Whether to build a new vocabulary

    Returns:
        encoded_padded: List of padded integer sequences
        vocab: The Vocabulary object
    """
    # Step 1: Clean
    cleaned = [clean_text(t) for t in texts]

    # Step 2: Tokenize
    tokenized = [tokenize(t) for t in cleaned]

    # Step 3: Build vocabulary
    if fit or vocab is None:
        vocab = Vocabulary(max_vocab_size=max_vocab_size)
        vocab.build(tokenized)

    # Step 4: Encode
    encoded = [vocab.encode(tokens) for tokens in tokenized]

    # Step 5: Pad
    padded = [pad_sequence(seq, max_len=max_seq_len) for seq in encoded]

    return padded, vocab


if __name__ == "__main__":
    # Quick test
    sample_texts = [
        "I hate those people, they should all disappear! @user #hate 🔥",
        "Good morning everyone! Hope you have a wonderful day ❤️",
        "Check out https://example.com for the latest news #trending",
    ]

    padded, vocab = preprocess_texts(sample_texts, max_seq_len=20)

    for i, (text, seq) in enumerate(zip(sample_texts, padded)):
        print(f"\n{'='*60}")
        print(f"Original:  {text}")
        print(f"Cleaned:   {clean_text(text)}")
        print(f"Tokens:    {tokenize(clean_text(text))}")
        print(f"Encoded:   {seq}")
        print(f"Decoded:   {vocab.decode(seq)}")
