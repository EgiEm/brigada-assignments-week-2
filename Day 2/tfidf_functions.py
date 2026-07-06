import math
from collections import Counter

def tf(word, document_tokens):
    """Llogarit Term Frequency (TF): sa herë shfaqet fjala / numri total i fjalëve në dokument"""
    if not document_tokens:
        return 0
    return document_tokens.count(word) / len(document_tokens)

def idf(word, all_documents_tokens):
    """Llogarit Inverse Document Frequency (IDF): log(numri total i dokumenteve / dokumentet që përmbajnë fjalën)"""
    N = len(all_documents_tokens)
    num_docs_with_word = sum(1 for doc in all_documents_tokens if word in doc)
    
    # Shmangim pjesëtimin me zero nëse fjala nuk ekziston fare
    if num_docs_with_word == 0:
        return 0
        
    return math.log(N / num_docs_with_word)

def tfidf(word, document_tokens, all_documents_tokens):
    """Llogarit peshën përfundimtare TF-IDF"""
    return tf(word, document_tokens) * idf(word, all_documents_tokens)