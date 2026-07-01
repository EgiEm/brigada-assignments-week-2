import csv  # Import standard CSV module to parse intents.csv files correctly
import os   # Import standard OS module to check if files exist

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
    # Check if the CSV dataset file exists at the specified path
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file {csv_path} not found.")

    rows = []            # List to store parsed text and label rows
    seen_texts = set()   # Set to store unique text inputs to detect duplicate rows (leakage)
    duplicates = []      # List to collect any duplicate lines found for debugging
    class_counts = {}    # Dictionary to keep track of the frequency of each intent class

    # Open the CSV file with utf-8 encoding to support German characters
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)          # Create a CSV reader object
        header = next(reader, None)    # Read and skip the first line (header)
        
        # Enumerate lines starting at line number 2 (since line 1 is the header)
        for line_num, row in enumerate(reader, start=2):
            # Skip empty lines or malformed rows that don't have text and label
            if not row or len(row) < 2:
                continue
            # Strip extra whitespaces from text and label fields
            text, label = row[0].strip(), row[1].strip()
            
            # If the current sentence has been seen before, it is a duplicate (leakage)
            if text in seen_texts:
                duplicates.append((text, line_num)) # Log the duplicated text and its line number
            seen_texts.add(text)                    # Add the sentence to our seen set
            
            # Increment the counter for this specific intent class
            class_counts[label] = class_counts.get(label, 0) + 1
            rows.append((text, label))              # Add the text and label tuple to our clean rows list

    # Print the validation report header
    print("=== DATASET BUILD REPORT ===")
    print(f"Total Rows: {len(rows)}")                    # Display the total number of lines processed
    print(f"Unique Sentences: {len(seen_texts)}")         # Display number of unique inputs

    # Report if there are duplicate sentences (data leakage between train/test partitions)
    if duplicates:
        print("\nWARNING: Leakage detected! Duplicate entries found:")
        for dup, line in duplicates:
            print(f" - '{dup}' (Line {line})")
    else:
        print("\nClean Check: Dataset is leakage-free (0 duplicates).")
        
    # Print the distribution counts for each intent class
    print("\nClass Distribution:")
    for label, count in sorted(class_counts.items()):
        print(f" - {label:18} : {count} rows")
        
    # Calculate the minimum and maximum class occurrences to verify class balancing
    min_count = min(class_counts.values()) if class_counts else 0
    max_count = max(class_counts.values()) if class_counts else 0
    
    # If the counts of all classes are equal and all 6 classes are present, it is perfectly balanced
    if min_count == max_count and len(class_counts) == 6:
        print("\nBalance Check: PERFECTLY BALANCED across all 6 intents.")
    # If the difference is at most 1, it is roughly balanced
    elif max_count - min_count <= 1:
        print("\nBalance Check: Roughly balanced (difference <= 1).")
    # Otherwise, warn the user about severe imbalance
    else:
        print("\nWARNING: Dataset is imbalanced!")
        
    return rows, class_counts # Return the parsed rows list and distribution dictionary

if __name__ == "__main__":
    # Multilingual German rows with English glosses:
    # ("Erinnere mich daran, Milch zu kaufen", "create_task")  # = remind me to buy milk
    # ("Ruf Mama an", "place_call")                            # = call mom
    # ("Stelle einen Timer auf 5 Minuten", "set_timer")        # = set a timer for 5 minutes
    
    # Run verification pipeline over intents.csv
    build_dataset("intents.csv")
