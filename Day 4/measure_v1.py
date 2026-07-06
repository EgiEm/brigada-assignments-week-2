import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. LOAD THE DATA
# ==========================================
# Reads your CSV file. Make sure intents.csv is in the same folder!
df = pd.read_csv("intents.csv")
X = df["text"]
y = df["label"]

# ==========================================
# PART A: THE BASELINE & SPLIT
# ==========================================
# Split data: 70% to practice (train), 30% hidden for the final exam (test)
# random_state=0 guarantees the exact same split every time you run it.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Build the pipeline: Text -> Numbers (TF-IDF) -> Brain (Logistic Regression)
clf = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))

# Train the model
clf.fit(X_train, y_train)

# Get predictions on the hidden test exam
preds = clf.predict(X_test)

# Calculate Overall Accuracy Score
overall_acc = accuracy_score(y_test, preds)

print("=" * 50)
print(f"OVERALL BASELINE ACCURACY: {overall_acc:.1%}")
print("=" * 50)

# Calculate Per-Class Accuracy breakdown
print("\nPER-CLASS ACCURACY BREAKDOWN:")
print(classification_report(y_test, preds))

# ==========================================
# PART B: THE CONFUSION MATRIX
# ==========================================
print("CONFUSION MATRIX:")
labels_order = sorted(list(y.unique()))
cm = confusion_matrix(y_test, preds, labels=labels_order)

# Convert confusion matrix to a clean readable DataFrame grid
cm_df = pd.DataFrame(cm, index=labels_order, columns=labels_order)
print(cm_df)
print("=" * 50)