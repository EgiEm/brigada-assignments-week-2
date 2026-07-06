import os
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Load the data
data_path = os.path.join(os.path.dirname(__file__), "intents.csv")
df = pd.read_csv(data_path)
X = df["text"]
y = df["label"]

# 2. Build the pipeline
clf = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))

# 3. Cross-Validation (5-fold)
cv_scores = cross_val_score(clf, X, y, cv=5)
mean_acc = np.mean(cv_scores)
std_acc = np.std(cv_scores)

print("=" * 60)
print("PART A: CROSS-VALIDATION RESULTS")
print(f"5-Fold CV Accuracies: {cv_scores}")
print(f"Mean Accuracy: {mean_acc:.4f}")
print(f"Standard Deviation (Spread): {std_acc:.4f}")
print(f"Formatted: {mean_acc:.1%} ± {std_acc:.1%}")
print("=" * 60)

# 4. Fit the model on all data and save it
clf.fit(X, y)
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(clf, f)
print(f"Model successfully saved to: {model_path}")
print("=" * 60)

# 5. Measure Latency
# Let's measure how long it takes to run 1000 single predictions
latencies = []
for _ in range(1000):
    start_time = time.perf_counter()
    clf.predict(["test query"])
    end_time = time.perf_counter()
    latencies.append((end_time - start_time) * 1000)  # to milliseconds

mean_latency = np.mean(latencies)
print("LATENCY MEASUREMENT")
print(f"Average prediction latency on CPU: {mean_latency:.4f} ms")
print("=" * 60)

# 6. Multilingual Stress Test
stress_test_cases = [
    # English sentences (expected to pass)
    {"text": "can you dial the office please", "gloss": "", "expected": "place_call"},
    {"text": "remind me to wash the car tomorrow morning", "gloss": "", "expected": "create_task"},
    {"text": "what is the capital of Japan", "gloss": "", "expected": "answer_question"},
    {"text": "please set a timer for 15 minutes", "gloss": "", "expected": "set_timer"},
    # Non-English sentences (expected to fail due to lexical gap)
    {"text": "Appelle mon frère", "gloss": "call my brother", "expected": "place_call"},
    {"text": "Planifie une tâche pour demain", "gloss": "schedule a task for tomorrow", "expected": "create_task"},
    {"text": "Wie ist die Hauptstadt von Japan", "gloss": "what is the capital of Japan", "expected": "answer_question"},
    {"text": "Stell einen Wecker auf sechs Uhr", "gloss": "set an alarm for six o'clock", "expected": "set_timer"},
]

print("MULTILINGUAL STRESS TEST RESULTS")
print(f"{'Sentence (Gloss)':<60} | {'Expected':<18} | {'Predicted':<18} | {'Result':<5}")
print("-" * 110)
for case in stress_test_cases:
    pred = clf.predict([case["text"]])[0]
    result = "PASS" if pred == case["expected"] else "FAIL"
    gloss_str = f" ({case['gloss']})" if case["gloss"] else ""
    sentence_display = f"\"{case['text']}\"{gloss_str}"
    print(f"{sentence_display:<60} | {case['expected']:<18} | {pred:<18} | {result:<5}")
print("=" * 60)
