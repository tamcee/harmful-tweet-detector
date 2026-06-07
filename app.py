"""
Interactive Web Demo — Context-Aware Harmful Sentiment Detection
================================================================
Flask web application to visualize harmful sentiment detection
on Twitter threads in real-time.
"""

import os
import sys
import json
import torch
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import clean_text, tokenize, Vocabulary, pad_sequence, preprocess_texts
from src.models import get_model
from src.dataset import create_dataloaders
from src.train import train_model
from src.evaluate import evaluate_model
from data.prepare_data import prepare_and_save

from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__, template_folder="templates", static_folder="static")

# ─── Global model state ────────────────────────────────────────────────────────
MODEL = None
VOCAB = None
CONFIG = {
    "max_vocab_size": 10000,
    "max_seq_len": 100,
    "embed_dim": 64,
    "hidden_dim": 64,
    "batch_size": 64,
    "epochs": 15,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "dropout": 0.5,
    "patience": 5,
    "seed": 42,
}

# ─── Sample threads for the demo ───────────────────────────────────────────────
SAMPLE_THREADS = [
    {
        "id": 1,
        "title": "Anti-Immigration Hate Thread",
        "author": "@toxic_user_42",
        "avatar_color": "#e74c3c",
        "tweets": [
            {"text": "I can't believe they let those people into our country. They are all disgusting!", "time": "2h"},
            {"text": "Someone should teach those immigrants a lesson, they don't belong here at all", "time": "2h"},
            {"text": "I hope every single one of them suffers for ruining everything we built", "time": "1h"},
            {"text": "Get rid of all foreigners, they're nothing but worthless trash to this nation", "time": "1h"},
        ],
    },
    {
        "id": 2,
        "title": "Cyberbullying Thread",
        "author": "@bully_anon",
        "avatar_color": "#e67e22",
        "tweets": [
            {"text": "You're such a worthless pathetic person, nobody even wants you around", "time": "5h"},
            {"text": "Go disappear you stupid loser, the world would be so much better without you", "time": "5h"},
            {"text": "Everyone hates you and you know it, you're absolutely deplorable garbage", "time": "4h"},
        ],
    },
    {
        "id": 3,
        "title": "Threats & Violent Rhetoric",
        "author": "@angry_poster",
        "avatar_color": "#9b59b6",
        "tweets": [
            {"text": "I swear I'm going to hurt the next person who disrespects me online", "time": "3h"},
            {"text": "Someone needs to destroy these terrible people who keep spreading lies about me", "time": "3h"},
            {"text": "Watch your back, I'll make you pay when you least expect it", "time": "2h"},
            {"text": "Violence is the only answer for dealing with protesters and activists", "time": "2h"},
            {"text": "Better sleep with one eye open because I'll teach you a lesson", "time": "1h"},
        ],
    },
    {
        "id": 4,
        "title": "AI & Technology Discussion",
        "author": "@tech_enthusiast",
        "avatar_color": "#3498db",
        "tweets": [
            {"text": "Just had an amazing discussion about machine learning ethics at the conference today!", "time": "6h"},
            {"text": "The speakers raised really important points about responsible AI development", "time": "6h"},
            {"text": "I think AI regulation is a fascinating topic that deserves more public attention", "time": "5h"},
            {"text": "Looking forward to implementing these new ideas in our research project next week", "time": "5h"},
        ],
    },
    {
        "id": 5,
        "title": "Positive Community Thread",
        "author": "@wholesome_vibes",
        "avatar_color": "#2ecc71",
        "tweets": [
            {"text": "Good morning everyone! Sending love and positive energy to all of you today", "time": "8h"},
            {"text": "Remember to take breaks and stay hydrated during your work day. You matter!", "time": "7h"},
            {"text": "Shoutout to everyone working hard on mental health awareness, you're truly inspiring", "time": "7h"},
            {"text": "Can't wait to learn more about community service initiatives this weekend", "time": "6h"},
        ],
    },
    {
        "id": 6,
        "title": "Civil Policy Debate",
        "author": "@policy_wonk",
        "avatar_color": "#1abc9c",
        "tweets": [
            {"text": "I respectfully disagree with the new healthcare reform proposal, here's my analysis", "time": "4h"},
            {"text": "While I understand the argument about data privacy, the evidence shows another story", "time": "4h"},
            {"text": "Let's have a civil discussion about economic inequality and find practical solutions", "time": "3h"},
            {"text": "Great points raised about public transportation, but consider this alternative perspective", "time": "3h"},
        ],
    },
    {
        "id": 7,
        "title": "Casual Daily Thread",
        "author": "@daily_thoughts",
        "avatar_color": "#f39c12",
        "tweets": [
            {"text": "Coffee and coding on a rainy day, honestly doesn't get better than this", "time": "10h"},
            {"text": "Just finished watching an amazing documentary about space exploration", "time": "9h"},
            {"text": "Weekend vibes! Time to relax and catch up on some reading finally", "time": "8h"},
            {"text": "Trying out a brand new recipe tonight, wish me luck everyone!", "time": "7h"},
        ],
    },
    {
        "id": 8,
        "title": "Mixed Content Thread",
        "author": "@mixed_opinions",
        "avatar_color": "#e74c3c",
        "tweets": [
            {"text": "Here are 5 tips for getting started with renewable energy at home", "time": "12h"},
            {"text": "I can't stand those awful people who keep posting garbage online all day", "time": "11h"},
            {"text": "Important update about climate change research that everyone should read", "time": "10h"},
            {"text": "All those pathetic protesters deserve to suffer for blocking traffic yesterday", "time": "9h"},
            {"text": "Really proud of my team for their incredible work on digital literacy programs", "time": "8h"},
        ],
    },
]


# ─── Lexicon for hybrid scoring ─────────────────────────────────────────────────

# Words/phrases scored by severity (0.0 to 1.0)
HARMFUL_LEXICON = {
    # Extreme violence / threats (0.8 - 1.0)
    "kill": 0.95, "murder": 0.95, "die": 0.85, "dead": 0.80,
    "shoot": 0.90, "stab": 0.90, "bomb": 0.90, "attack": 0.80,
    "destroy": 0.75, "eliminate": 0.80, "hurt": 0.70, "beat": 0.70,
    "punch": 0.70, "slap": 0.65, "fight": 0.55, "smash": 0.70,
    "burn": 0.75, "torture": 0.90, "strangle": 0.90, "choke": 0.80,
    "threaten": 0.75, "suffer": 0.70, "punish": 0.65, "revenge": 0.65,
    "violent": 0.70, "violence": 0.70, "weapon": 0.75,

    # Hate speech / slurs (0.6 - 0.9)
    "hate": 0.65, "disgusting": 0.60, "pathetic": 0.55, "worthless": 0.60,
    "trash": 0.55, "scum": 0.70, "garbage": 0.55, "deplorable": 0.55,
    "vile": 0.65, "awful": 0.45, "terrible": 0.40, "stupid": 0.45,
    "idiot": 0.50, "moron": 0.55, "loser": 0.50, "dumb": 0.40,
    "ugly": 0.45, "fat": 0.40, "freak": 0.55, "creep": 0.50,
    "racist": 0.60, "sexist": 0.60, "bigot": 0.60,

    # Cyberbullying indicators (0.5 - 0.8)
    "bully": 0.60, "harass": 0.65, "stalk": 0.70, "doxx": 0.80,
    "shame": 0.50, "humiliate": 0.60, "embarrass": 0.45,
    "nobody wants": 0.65, "nobody cares": 0.60, "everyone hates": 0.70,
    "shut up": 0.50, "go away": 0.35, "disappear": 0.55,
    "deserve": 0.40, "suffer": 0.65,

    # Dehumanization (0.6 - 0.8)
    "animal": 0.35, "vermin": 0.75, "parasite": 0.75, "cockroach": 0.80,
    "plague": 0.65, "disease": 0.45, "cancer": 0.50, "poison": 0.55,
    "rid of": 0.55, "get rid": 0.55, "wipe out": 0.75, "cleanse": 0.70,

    # Aggressive intent
    "swear": 0.40, "gonna": 0.20, "watch your back": 0.75,
    "pay for": 0.50, "lesson": 0.35, "regret": 0.35,
}

SAFE_LEXICON = {
    # Positive emotions
    "love": 0.70, "happy": 0.65, "grateful": 0.70, "thankful": 0.65,
    "amazing": 0.60, "wonderful": 0.65, "beautiful": 0.60, "awesome": 0.55,
    "great": 0.50, "good": 0.40, "nice": 0.40, "kind": 0.55,
    "caring": 0.60, "supportive": 0.60, "inspiring": 0.60, "proud": 0.55,
    "excited": 0.55, "enjoy": 0.50, "fun": 0.45, "hope": 0.50,
    "encourage": 0.55, "celebrate": 0.55, "appreciate": 0.60,

    # Community / cooperation
    "together": 0.50, "community": 0.55, "team": 0.45, "share": 0.40,
    "help": 0.50, "support": 0.50, "volunteer": 0.55, "donate": 0.50,
    "collaborate": 0.55, "contribute": 0.50, "unity": 0.55,

    # Civil discourse
    "discuss": 0.45, "respectfully": 0.55, "perspective": 0.45,
    "consider": 0.40, "understand": 0.45, "research": 0.40,
    "learn": 0.45, "education": 0.50, "interesting": 0.40,
    "disagree": 0.30, "opinion": 0.30, "debate": 0.35,
    "analysis": 0.40, "evidence": 0.40, "data": 0.35,

    # Everyday positive
    "morning": 0.35, "coffee": 0.30, "weekend": 0.35, "relax": 0.40,
    "recipe": 0.30, "movie": 0.30, "book": 0.30, "music": 0.35,
    "friends": 0.45, "family": 0.45, "recommend": 0.40,
    "congratulations": 0.55, "welcome": 0.45, "please": 0.30, "thank": 0.50,
}


def lexicon_score(text):
    """
    Score text using harmful/safe lexicons.
    Returns a value between 0.0 (safe) and 1.0 (harmful).
    """
    words = text.lower().split()
    text_lower = text.lower()

    harmful_score = 0.0
    safe_score = 0.0
    harmful_hits = 0
    safe_hits = 0

    # Check multi-word phrases first
    for phrase, score in HARMFUL_LEXICON.items():
        if " " in phrase and phrase in text_lower:
            harmful_score += score
            harmful_hits += 1

    for phrase, score in SAFE_LEXICON.items():
        if " " in phrase and phrase in text_lower:
            safe_score += score
            safe_hits += 1

    # Check single words
    for word in words:
        clean_word = word.strip(".,!?;:'\"()[]{}").lower()
        if clean_word in HARMFUL_LEXICON:
            harmful_score += HARMFUL_LEXICON[clean_word]
            harmful_hits += 1
        if clean_word in SAFE_LEXICON:
            safe_score += SAFE_LEXICON[clean_word]
            safe_hits += 1

    # Normalize: average score per hit, then balance harmful vs safe
    avg_harmful = (harmful_score / harmful_hits) if harmful_hits > 0 else 0.0
    avg_safe = (safe_score / safe_hits) if safe_hits > 0 else 0.0

    # Weight by number of hits (more hits = more confidence)
    harmful_weight = min(harmful_hits * 0.3, 1.0)
    safe_weight = min(safe_hits * 0.3, 1.0)

    # Combine into a single score: 0.0 (safe) to 1.0 (harmful)
    if harmful_hits == 0 and safe_hits == 0:
        return 0.3  # Neutral default for unknown text
    elif harmful_hits == 0:
        return max(0.05, 0.3 - avg_safe * safe_weight * 0.3)
    elif safe_hits == 0:
        return min(0.95, 0.3 + avg_harmful * harmful_weight * 0.7)
    else:
        # Both present: blend
        net = (avg_harmful * harmful_weight) - (avg_safe * safe_weight)
        return max(0.05, min(0.95, 0.5 + net * 0.5))


def predict_tweet(text):
    """
    Hybrid prediction: combines RNN model output with lexicon-based scoring.
    This produces nuanced probability values instead of extreme 0%/100%.
    """
    global MODEL, VOCAB

    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    encoded = VOCAB.encode(tokens)

    # Compute lexicon score (always available, even for OOV text)
    lex_score = lexicon_score(text)

    # Check how many words the model actually recognizes
    known_words = sum(1 for idx in encoded if idx not in [0, 1])
    total_words = max(len(tokens), 1)
    vocab_coverage = known_words / total_words  # 0.0 to 1.0

    if known_words == 0:
        # No recognized words: rely entirely on lexicon
        probability = lex_score
    else:
        # Get model prediction with temperature scaling
        padded = pad_sequence(encoded, max_len=CONFIG["max_seq_len"])
        tensor = torch.tensor([padded], dtype=torch.long)

        MODEL.eval()
        with torch.no_grad():
            output = MODEL(tensor)
            model_prob = output.item()

        # Temperature scaling: soften extreme predictions
        # Map through logit space with temperature to prevent saturation
        import math
        eps = 1e-7
        logit = math.log((model_prob + eps) / (1 - model_prob + eps))
        temperature = 3.0  # Higher = softer probabilities
        scaled_prob = 1 / (1 + math.exp(-logit / temperature))

        # Blend model prediction with lexicon based on vocabulary coverage
        # High coverage = trust model more; Low coverage = trust lexicon more
        model_weight = vocab_coverage * 0.7  # Max 70% model
        lex_weight = 1.0 - model_weight       # Rest from lexicon

        probability = (scaled_prob * model_weight) + (lex_score * lex_weight)

    # Clamp to avoid exact 0.0 or 1.0
    probability = max(0.02, min(0.98, probability))
    label = 1 if probability >= 0.5 else 0

    return {
        "label": label,
        "label_text": "harmful" if label == 1 else "non-harmful",
        "confidence": round(probability if label == 1 else 1 - probability, 4),
        "probability": round(probability, 4),
    }


def train_and_load_model():
    """Load the best model and vocab from disk, or train if not found."""
    global MODEL, VOCAB
    import pandas as pd
    
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    model_path = os.path.join(results_dir, "best_model.pth")
    vocab_path = os.path.join(results_dir, "vocab.json")
    
    if os.path.exists(model_path) and os.path.exists(vocab_path):
        print("\n⚡ Loading pre-trained model and vocabulary from disk...")
        VOCAB = Vocabulary.load(vocab_path)
        
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model_name = checkpoint['model_name']
        config = checkpoint.config if hasattr(checkpoint, 'config') else checkpoint.get('config', CONFIG)
        
        MODEL = get_model(model_name, vocab_size=len(VOCAB), embed_dim=config["embed_dim"], dropout=config["dropout"])
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        MODEL.eval()
        print(f"✅ Loaded {model_name.upper()} model.")
        return

    print("\n🔧 Preparing models for web demo...")

    # Generate data
    df = prepare_and_save(n_samples=5000)
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # Preprocess
    padded_sequences, vocab = preprocess_texts(
        texts,
        max_vocab_size=CONFIG["max_vocab_size"],
        max_seq_len=CONFIG["max_seq_len"],
    )
    VOCAB = vocab

    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        padded_sequences, labels,
        batch_size=CONFIG["batch_size"],
        seed=CONFIG["seed"],
    )

    best_model = None
    best_f1 = -1
    best_name = ""

    for model_name in ["lstm", "gru"]:
        print(f"\n{'═' * 50}")
        print(f"  Training {model_name.upper()} + ADAM")
        print(f"{'═' * 50}")

        model = get_model(model_name, vocab_size=len(vocab), embed_dim=CONFIG["embed_dim"], dropout=CONFIG["dropout"])
        train_model(
            model, train_loader, val_loader,
            optimizer_name="adam",
            lr=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"],
            epochs=CONFIG["epochs"],
            patience=CONFIG["patience"],
        )

        # Evaluate
        metrics, _, _ = evaluate_model(model, test_loader)
        print(f"  ✅ {model_name.upper()} — Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = model
            best_name = model_name

    print(f"\n🏆 Best model: {best_name.upper()} (F1: {best_f1:.4f})")
    MODEL = best_model
    MODEL.eval()


# ─── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/threads")
def get_threads():
    """Return all sample threads."""
    return jsonify(SAMPLE_THREADS)


@app.route("/api/analyze", methods=["POST"])
def analyze_thread():
    """Analyze a thread of tweets."""
    data = request.json
    tweets = data.get("tweets", [])

    results = []
    for tweet_text in tweets:
        result = predict_tweet(tweet_text)
        results.append(result)

    harmful_count = sum(1 for r in results if r["label"] == 1)
    total = len(results)
    thread_harmful = harmful_count > total / 2

    return jsonify({
        "tweet_results": results,
        "summary": {
            "harmful_count": harmful_count,
            "total": total,
            "thread_label": "harmful" if thread_harmful else "non-harmful",
            "threat_level": round(harmful_count / total, 2) if total > 0 else 0,
        },
    })


@app.route("/api/analyze-single", methods=["POST"])
def analyze_single():
    """Analyze a single tweet."""
    data = request.json
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Empty text"}), 400
    result = predict_tweet(text)
    return jsonify(result)


@app.route("/results/<path:filename>")
def serve_result_file(filename):
    """Serve result images and files."""
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    return send_from_directory(results_dir, filename)


@app.route("/api/metrics")
def get_metrics():
    """Return evaluation metrics from the results table."""
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    table_path = os.path.join(results_dir, "results_table.txt")

    metrics = []
    best_combo = ""

    if os.path.exists(table_path):
        with open(table_path, "r") as f:
            content = f.read()

        for line in content.strip().split("\n"):
            # Match data lines like: LSTM             ADAM             1.0000 ...
            match = re.match(
                r"^(\w+)\s+(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
                line.strip(),
            )
            if match:
                metrics.append({
                    "model": match.group(1),
                    "optimizer": match.group(2),
                    "accuracy": float(match.group(3)),
                    "precision": float(match.group(4)),
                    "recall": float(match.group(5)),
                    "f1": float(match.group(6)),
                })

        # Extract best combination
        best_match = re.search(r"Best combination: (.+)", content)
        if best_match:
            best_combo = best_match.group(1)

    # List available chart images
    charts = []
    if os.path.exists(results_dir):
        for fname in sorted(os.listdir(results_dir)):
            if fname.endswith(".png"):
                charts.append(fname)

    return jsonify({
        "metrics": metrics,
        "best_combination": best_combo,
        "charts": charts,
    })


# ─── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_and_load_model()
    print("\n🌐 Starting web server at http://localhost:5001")
    app.run(debug=False, port=5001, host="0.0.0.0")
