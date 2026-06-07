"""
Data Preparation Module
=======================
Processes a real hate speech dataset (Davidson et al.) for harmful sentiment detection.
Maps:
- Hate Speech (0) / Offensive (1) -> Harmful (1)
- Neither (2) -> Non-harmful (0)
"""

import pandas as pd
import os
import re

def clean_tweet(text):
    """Clean standard Twitter artifacts."""
    # Remove RT @username:
    text = re.sub(r'\bRT\s+@\w+:', '', text)
    # Remove @usernames
    text = re.sub(r'@\w+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove HTML entities like &amp;
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_and_save(output_dir="data", n_samples=None):
    """Load the Davidson dataset, clean, balance, and save."""
    os.makedirs(output_dir, exist_ok=True)
    input_path = os.path.join(output_dir, "labeled_data.csv")
    output_path = os.path.join(output_dir, "twitter_threads.csv")

    if not os.path.exists(input_path):
        import urllib.request
        import ssl
        print("📥 Downloading hate speech dataset...")
        url = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
        context = ssl._create_unverified_context()
        urllib.request.urlretrieve(url, input_path, context=context)

    print("📦 Loading and processing real dataset...")
    df = pd.read_csv(input_path)
    
    # Map classes: 0 (Hate) and 1 (Offensive) -> 1 (Harmful)
    # 2 (Neither) -> 0 (Non-harmful)
    df['label'] = df['class'].apply(lambda x: 1 if x in [0, 1] else 0)
    
    # Clean text
    df['text'] = df['tweet'].apply(clean_tweet)
    
    # Drop empties
    df = df[df['text'].str.len() > 5]
    
    # We want a moderately balanced dataset to avoid trivial convergence
    # There are ~20k harmful and 4k non-harmful. Let's sample down the harmful class.
    df_harmful = df[df['label'] == 1]
    df_safe = df[df['label'] == 0]
    
    n_safe = len(df_safe)
    harmful_subset = df_harmful.sample(n=min(len(df_harmful), int(n_safe * 1.5)), random_state=42)
    
    df_balanced = pd.concat([harmful_subset, df_safe]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    if n_samples and n_samples < len(df_balanced):
        df_balanced = df_balanced.sample(n=n_samples, random_state=42).reset_index(drop=True)
        
    # Save the required columns
    df_final = df_balanced[['text', 'label']]
    df_final.to_csv(output_path, index=False)
    
    print(f"✅ Dataset saved to {output_path}")
    print(f"   Total samples: {len(df_final)}")
    print(f"   Harmful (1):   {(df_final['label'] == 1).sum()}")
    print(f"   Non-harmful (0): {(df_final['label'] == 0).sum()}")
    print(f"\n📝 Sample tweets:")
    for _, row in df_final.head(5).iterrows():
        label_str = "🔴 HARMFUL" if row["label"] == 1 else "🟢 NON-HARMFUL"
        print(f"   [{label_str}] {row['text'][:80]}...")
        
    return df_final

if __name__ == "__main__":
    prepare_and_save()
