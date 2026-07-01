import csv
import os

# =====================================================================
# CAVEAT & REGULATION
# Note: never strip Swiss-German ß to ss on classifier input — 
# that normalisation is for generated output only.
# =====================================================================

def build_dataset(csv_path="intents.csv"):
    """
    Loads and validates the dataset from intents.csv.
    Performs leakage checking (no duplicate inputs) and balance checking.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file {csv_path} not found.")

    rows = []
    seen_texts = set()
    duplicates = []
    class_counts = {}

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None) # Skip header
        
        for line_num, row in enumerate(reader, start=2):
            if not row or len(row) < 2:
                continue
            text, label = row[0].strip(), row[1].strip()
            
            # Check for leakage/duplicates
            if text in seen_texts:
                duplicates.append((text, line_num))
            seen_texts.add(text)
            
            # Count per class
            class_counts[label] = class_counts.get(label, 0) + 1
            rows.append((text, label))

    print("=== DATASET BUILD REPORT ===")
    print(f"Total Rows: {len(rows)}")
    print(f"Unique Sentences: {len(seen_texts)}")
    
    if duplicates:
        print("\nWARNING: Leakage detected! Duplicate entries found:")
        for dup, line in duplicates:
            print(f" - '{dup}' (Line {line})")
    else:
        print("\nClean Check: Dataset is leakage-free (0 duplicates).")
        
    print("\nClass Distribution:")
    for label, count in sorted(class_counts.items()):
        print(f" - {label:18} : {count} rows")
        
    # Check balance
    min_count = min(class_counts.values()) if class_counts else 0
    max_count = max(class_counts.values()) if class_counts else 0
    
    if min_count == max_count and len(class_counts) == 6:
        print("\nBalance Check: PERFECTLY BALANCED across all 6 intents.")
    elif max_count - min_count <= 1:
        print("\nBalance Check: Roughly balanced (difference <= 1).")
    else:
        print("\nWARNING: Dataset is imbalanced!")
        
    return rows, class_counts

if __name__ == "__main__":
    # Multilingual German rows with English glosses:
    # ("Erinnere mich daran, Milch zu kaufen", "create_task")  # = remind me to buy milk
    # ("Ruf Mama an", "place_call")                            # = call mom
    # ("Stelle einen Timer auf 5 Minuten", "set_timer")        # = set a timer for 5 minutes
    
    # Run verification pipeline over intents.csv
    build_dataset("intents.csv")
