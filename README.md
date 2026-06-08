# harmful-tweet-detector

Context-aware harmful sentiment detection in Twitter threads using bidirectional LSTM and GRU networks, with an interactive Flask web demo for real-time analysis.

## Stack

- Python 3.9+
- PyTorch (LSTM & GRU models)
- Flask (web demo)
- NLTK (text preprocessing)
- scikit-learn (evaluation metrics, cross-validation)
- matplotlib / seaborn (visualization)
- Dataset: Davidson et al. Hate Speech & Offensive Language (CSV) – 25k tweets labeled for harmful content
- Dataset: [Davidson et al. Hate Speech & Offensive Language](https://github.com/t-davidson/hate-speech-and-offensive-language)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

**Train models & generate evaluation plots:**

```bash
python main.py
```

Runs the full pipeline: data prep → preprocessing → 5-fold CV → LSTM vs GRU training → confusion matrices, comparison charts, and results table saved to `results/`.

**Launch the web demo:**

```bash
python app.py
```

Opens at `http://localhost:5001`. Loads the pre-trained best model and lets you analyze sample threads or type custom tweets for real-time harmful/non-harmful classification with confidence scores.

## Notes

- The model uses a hybrid approach: RNN output is blended with a lexicon-based scorer weighted by vocabulary coverage, which prevents extreme predictions on out-of-vocabulary text.
- On first run, `main.py` downloads the Davidson dataset (~2.5 MB) automatically. Subsequent runs use the cached CSV.
- The pre-trained model checkpoint (`results/best_model.pth`) is included so `app.py` can run without retraining.
