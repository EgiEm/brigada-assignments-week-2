# Week 2 Day 4: Measure v1 & Error Analysis

## Part A — The Baseline Number
We ran a reproducible 70/30 split on `intents.csv` (`test_size=0.3`, `random_state=0`) and trained a TF-IDF vectorizer paired with a Logistic Regression classifier (`max_iter=1000`).

*   **Overall Baseline Accuracy:** **46.2%** (6/13 correct predictions on the held-out test slice).

### Per-Class Accuracy Breakdown:
| Intent Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **answer_question** | 0.25 | 1.00 | 0.40 | 1 |
| **create_task** | 0.20 | 1.00 | 0.33 | 1 |
| **out_of_scope** | 0.00 | 0.00 | 0.00 | 3 |
| **place_call** | 1.00 | 0.50 | 0.67 | 2 |
| **save_memory** | 1.00 | 1.00 | 1.00 | 2 |
| **set_timer** | 1.00 | 0.25 | 0.40 | 4 |

---

## Part B — The Confusion Matrix
The confusion matrix over the 13 held-out predictions is shown below:

```text
                 answer_question  create_task  out_of_scope  place_call  save_memory  set_timer
answer_question                1            0             0           0            0          0
create_task                    0            1             0           0            0          0
out_of_scope                   1            2             0           0            0          0
place_call                     0            1             0           1            0          0
save_memory                    0            0             0           0            2          0
set_timer                      2            1             0           0            0          1
```

### Most-Confused Intent Pairs:
1.  **`out_of_scope` confused with `create_task` (2 counts) & `answer_question` (1 count):**
    *   **Reason:** The `out_of_scope` class is a "catch-all" category containing very diverse queries (e.g., *“play some jazz music”*, *“turn off the living room lights”*, *“how long will it take to get to work”*). These queries do not share any cohesive vocabulary. Consequently, TF-IDF + Logistic Regression fails to form a distinct centroid for `out_of_scope`, causing the model to misclassify them into classes with overlapping stop words or common verbs (like *"to"*, *"me"* in `create_task`, or *"what"* in `answer_question`).
2.  **`set_timer` confused with `answer_question` (2 counts) & `create_task` (1 count):**
    *   **Reason:** Simple TF-IDF matching relies heavily on exact keyword frequencies. A test sentence like *“Stelle einen Timer auf 5 Minuten”* or *“timer for thirty seconds please”* may get misclassified due to a lack of shared English vocabulary keywords in the small training set, or overlapping patterns like numeric values and prepositions.

---

## Part C — Predicted Fixes to Raise Accuracy
1.  **Introduce an Embedding-Based Router (Semantic Search):**
    Using pre-trained semantic sentence embeddings (like `all-MiniLM-L6-v2`) instead of pure TF-IDF will allow the model to recognize synonyms and intent context (e.g., *“stopwatch”*, *“timer”*, *“countdown”*) even without literal keyword matches. This is predicted to resolve the `set_timer` misclassifications.
2.  **Increase Diversity and Volume of `out_of_scope` Training Examples:**
    Adding more training examples to `out_of_scope` containing common out-of-scope verbs and nouns will help the classifier establish a broader boundary for what *doesn't* fit the core five intents.
