import os
import csv
# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

def load_data(csv_path):
    texts = []
    labels = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip header
        for row in reader:
            if not row or len(row) < 2:
                continue
            texts.append(row[0].strip())
            labels.append(row[1].strip())
    return texts, labels

def main():
    # File paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "Week 2 Day 1", "intents.csv")
    csv_path = os.path.normpath(csv_path)

    print(f"Loading data from {csv_path}...")
    texts, labels = load_data(csv_path)
    print(f"Loaded {len(texts)} samples belonging to {len(set(labels))} unique classes.")

    # Build the pipeline
    clf = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
    
    # Train
    clf.fit(texts, labels)
    print("Model trained successfully.")

    # Confirm model can predict training sentences
    train_preds = clf.predict(texts)
    accuracy = np.mean(train_preds == labels)
    print(f"Training Accuracy: {accuracy * 100:.2f}%")

    # Define the 10 probe sentences
    probe_sentences = [
        # German examples (with inline English gloss)
        ("Ruf meinen Bruder an", "Ruf meinen Bruder an (call my brother)"),
        ("Erinnere mich um 6", "Erinnere mich um 6 (remind me at 6)"),
        
        # Boundary examples
        ("remind me to call John", "remind me to call John"),
        ("set a timer to buy milk", "set a timer to buy milk"),
        
        # Other unseen spread across intents
        ("who is the president of the United States", "who is the president of the United States"),
        ("what is the weather in Pristina", "what is the weather in Pristina"),
        ("write down that the front door code is 1234", "write down that the front door code is 1234"),
        ("start a stopwatch for twenty minutes", "start a stopwatch for twenty minutes"),
        ("dial the police department", "dial the police department"),
        ("can you play some rock music", "can you play some rock music")
    ]

    print("\n--- Probing Results ---")
    print(f"{'Sentence':<50} | {'Predicted Label':<18} | {'Probability':<12}")
    print("-" * 88)
    
    for sentence, display_text in probe_sentences:
        pred_label = clf.predict([sentence])[0]
        probs = clf.predict_proba([sentence])[0]
        top_prob = max(probs)
        print(f"{display_text:<50} | {pred_label:<18} | {top_prob:.4f}")

if __name__ == "__main__":
    main()

# commit message update
