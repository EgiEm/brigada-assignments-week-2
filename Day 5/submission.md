# Week 2 Day 5: Ship v1 + Model Card

This document presents the Week 2 deliverables: our cross-validation analysis, a multilingual stress-test evaluation, a saved model file, and the official **v1 Model Card** contract that Week 3 will inherit.

---

## Part A — Cross-Validation & Baseline Comparison

We evaluated our TF-IDF + Logistic Regression pipeline on the entire `intents.csv` dataset using 5-fold cross-validation (`cv=5`).

*   **5-Fold CV Accuracy Scores:** `[44.4%, 44.4%, 75.0%, 37.5%, 50.0%]`
*   **Cross-Validated Mean Accuracy:** **50.3%**
*   **Cross-Validated Spread (Std Dev):** **± 13.0%** (50.3% ± 13.0%)
*   **Day-4 Single-Split Baseline (for comparison):** **46.2%**

### Analysis of Day-4 Baseline
Our Day-4 single-split accuracy of **46.2%** was slightly **unlucky**, as it sits below the cross-validated mean of **50.3%**. However, because the standard deviation spread is relatively wide (13.0%, ranging from 37.3% to 63.3%), this variation is a classic symptom of a small dataset (42 rows). A single split's score is highly sensitive to which specific sentences land in the test set. Cross-validation averages out this noise, giving us a more stable, trustworthy baseline.

---

## Part B — Multilingual Stress Test

We evaluated the fully trained `v1` classifier against a test suite of 4 English and 4 non-English (French/German) sentences to check if intent could be successfully routed across languages.

| Sentence | English Gloss (if applicable) | Expected Intent | Predicted Intent | Result |
| :--- | :--- | :--- | :--- | :---: |
| "can you dial the office please" | - | `place_call` | `place_call` | **PASS** |
| "remind me to wash the car tomorrow morning" | - | `create_task` | `create_task` | **PASS** |
| "what is the capital of Japan" | - | `answer_question` | `answer_question` | **PASS** |
| "please set a timer for 15 minutes" | - | `set_timer` | `set_timer` | **PASS** |
| "Appelle mon frère" | *call my brother* | `place_call` | `out_of_scope` | **FAIL** |
| "Planifie une tâche pour demain" | *schedule a task for tomorrow* | `create_task` | `out_of_scope` | **FAIL** |
| "Wie ist die Hauptstadt von Japan" | *what is the capital of Japan* | `answer_question` | `out_of_scope` | **FAIL** |
| "Stell einen Wecker auf sechs Uhr" | *set an alarm for six o'clock* | `set_timer` | `set_timer` | **PASS** |

### Explanation of the Cross-Lingual Gap
TF-IDF is a **lexical** method; it maps text into vectors based purely on the literal frequency of words seen during training. Because the training set is predominantly English, non-English terms (such as French *"Appelle"* vs. English *"call"*) reside in entirely different dimensions in the vocabulary space. As a result, the classifier fails to link them to the correct intent, falling back to `out_of_scope`. 
*Note on the "Stell..." pass:* It got lucky because the training set contained German words like `"einen"` and `"auf"` in the `set_timer` class, showing how lexical matching relies on superficial keyword correlations rather than true semantic understanding.

---

## Part C — The v1 Model Card

This is the baseline contract. Week 3's embedding-based router must beat the numbers and address the failure modes listed below.

```text
================================================================================
                                v1 MODEL CARD
================================================================================

1. DATASET
   - Total Rows: 42
   - Number of Intents: 6
   - Class Balance: Perfectly balanced (exactly 7 rows per intent class):
     - create_task: 7 rows
     - place_call: 7 rows
     - answer_question: 7 rows
     - save_memory: 7 rows
     - set_timer: 7 rows
     - out_of_scope: 7 rows

2. ACCURACY
   - Day-4 Single-Split Baseline: 46.2%
   - Part-A Cross-Validated Score: 50.3% ± 13.0%

3. TOP-3 FAILURE MODES
   - Multilingual Lexical Mismatch: Fails on non-English queries that do not
     share literal vocabulary terms with the training set.
   - Out-of-Scope Confusion: Diverse queries under "out_of_scope" do not share a
     cohesive vocabulary, leaking into other classes or absorbing unseen inputs.
   - Lexical Synonym Gap: Inability to generalise to unseen vocabulary synonyms 
     (e.g., "stopwatch", "alarm", "countdown") within the same language.

4. LATENCY NOTE
   - v1 (TF-IDF + Logistic Regression) runs predictions in milliseconds on CPU
     (measured average: ~0.91 ms). It requires no GPU resource.
   - In contrast, OXODIN's production LLM router (Qwen3-14B-AWQ) takes ~3.4s, 
     making v1 more than 3,700x faster, though significantly less robust.

5. WHAT V2 SHOULD IMPROVE
   - Transitioning from lexical TF-IDF to semantic sentence embeddings (bge-m3)
     will represent sentences by their conceptual meaning rather than exact
     words, resolving the multilingual and synonym gaps.
================================================================================
```

---

## Code & Artifact Reference

*   **Trained Model Artifact:** [model.pkl](file:///c:/Users/beKs/Desktop/Brigada/Week%202%20Day%205/model.pkl)
*   **Training and Evaluation Script:** [train_and_save_v1.py](file:///c:/Users/beKs/Desktop/Brigada/Week%202%20Day%205/train_and_save_v1.py)
*   **Dataset:** [intents.csv](file:///c:/Users/beKs/Desktop/Brigada/Week%202%20Day%205/intents.csv)

The model can be reloaded and run without retraining using the snippet below:
```python
import pickle

with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Predict intent
intent = loaded_model.predict(["Stell einen Wecker auf sechs Uhr"])[0]
print(f"Predicted Intent: {intent}")
```
