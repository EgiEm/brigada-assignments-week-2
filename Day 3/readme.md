# Week 2 Day 3: Train v1 and Probe It

This folder contains the training pipeline and probing analysis for our v1 intent classifier.

## Part A: Training Code

Below is the core training logic implemented in [train_and_probe.py](file:///c:/Users/beKs/Desktop/Brigada/Week%202%20Day%203/train_and_probe.py):

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Build the pipeline
clf = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))

# Train the model on texts and labels loaded from intents.csv
clf.fit(texts, labels)
```

The pipeline uses `TfidfVectorizer` to convert training sentences into numerical TF-IDF feature matrices and fits a `LogisticRegression` classifier on them.

---

## Part B: 10-Sentence Probe Table

The model was probed with 10 unseen sentences. Here are the results:

| Sentence | Predicted Label | Top Probability |
| :--- | :--- | :--- |
| **Ruf meinen Bruder an** *(call my brother)* | `place_call` | 0.2848 |
| **Erinnere mich um 6** *(remind me at 6)* | `create_task` | 0.2315 |
| **remind me to call John** | `create_task` | 0.3265 |
| **set a timer to buy milk** | `create_task` | 0.2681 |
| **who is the president of the United States** | `answer_question` | 0.3480 |
| **what is the weather in Pristina** | `answer_question` | 0.2932 |
| **write down that the front door code is 1234** | `save_memory` | 0.3440 |
| **start a stopwatch for twenty minutes** | `set_timer` | 0.4046 |
| **dial the police department** | `place_call` | 0.3153 |
| **can you play some rock music** | `out_of_scope` | 0.3429 |

---

## Part C: Failure Analysis & Hypotheses

Here are three clear failure/low-confidence predictions and the hypotheses explaining them:

1. **Sentence:** `"set a timer to buy milk"`
   - **Predicted Label:** `create_task` (Confidence: `0.2681`) | **Real Intent:** `set_timer`
   - **Hypothesis:** Lexical overlap where specific training tokens like "buy" and "milk" (previously seen only in `create_task` examples) bias the sparse TF-IDF representation away from the target intent `set_timer` despite the presence of "set a timer".

2. **Sentence:** `"what is the weather in Pristina"`
   - **Predicted Label:** `answer_question` (Confidence: `0.2932`) | **Real Intent:** `out_of_scope`
   - **Hypothesis:** Generic query structure tokens ("what", "is", "the") that occur frequently in `answer_question` training templates dominate the TF-IDF representation, causing the model to misclassify this sentence since it has no semantic understanding of the unseen words "weather" and "Pristina".

3. **Sentence:** `"remind me to call John"`
   - **Predicted Label:** `create_task` (Confidence: `0.3265`) | **Real Intent:** `place_call` (or boundary case)
   - **Hypothesis:** The multiple prefix tokens "remind me to" (strongly associated with `create_task` in training) heavily bias the classification, completely overriding the single token "call" which is shared across both intents.
