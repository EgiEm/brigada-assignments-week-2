import csv
import re
import os
from tfidf_functions import tfidf

def clean_and_tokenize(text):
    """Pastron tekstin nga shenjat e pikësimit dhe e kthen në shkronja të vogla"""
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def create_dummy_csv():
    """Krijon një intents.csv automatikisht nëse nuk ekziston"""
    data = [
        ["text", "label"],
        ["remind me to buy milk tomorrow", "create_task"],
        ["set an alarm for 7 am", "create_task"],
        ["make a note that i spent 20 dollars", "save_memory"],
        ["remember that my car keys are on the table", "save_memory"],
        ["what is the weather like today", "get_weather"],
        ["will it rain in pristina this weekend", "get_weather"],
        ["play some relaxing music on spotify", "play_media"],
        ["turn up the volume please", "play_media"],
        ["book a flight to tirana next week", "book_travel"],
        ["find a cheap hotel in saranda", "book_travel"],
        ["tell me a funny joke to make me laugh", "entertainment"],
        ["do you know any good riddles", "entertainment"]
    ]
    with open('intents.csv', mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print("[INFO] Skedari 'intents.csv' u krijua automatikisht me të dhëna shembull!")

def main():
    # Kontrollojmë nëse ekziston intents.csv, nëse jo e krijojmë
    if not os.path.exists('intents.csv'):
        create_dummy_csv()

    intents_data = {}
    
    with open('intents.csv', mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader, None) # Kalojmë header-in
        
        for row in reader:
            if len(row) >= 2:
                sentence = row[0]
                intent = row[1].strip()
                
                if intent not in intents_data:
                    intents_data[intent] = []
                intents_data[intent].append(sentence)

    intent_names = list(intents_data.keys())
    all_documents_tokens = []
    intent_to_tokens = {}

    for intent in intent_names:
        combined_text = " ".join(intents_data[intent])
        tokens = clean_and_tokenize(combined_text)
        intent_to_tokens[intent] = tokens
        all_documents_tokens.append(tokens)

    print("\n=== PART A: TOP 5 TERMS PER INTENT ===")
    for intent in intent_names:
        current_tokens = intent_to_tokens[intent]
        unique_words = set(current_tokens)
        
        word_scores = {}
        for word in unique_words:
            score = tfidf(word, current_tokens, all_documents_tokens)
            word_scores[word] = score
            
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        top_5 = sorted_words[:5]
        
        print("")
        print(f"Intent: {intent}")
        print("-" * 30)
        print(f"{'Term':<15} | {'Weight':<10}")
        print("-" * 30)
        for term, weight in top_5:
            print(f"{term:<15} | {weight:.3f}")
        print("-" * 30)

if __name__ == "__main__":
    main()