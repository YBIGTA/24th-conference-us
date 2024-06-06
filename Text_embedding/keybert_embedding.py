# keybert_embedding.py
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import torch
import argparse

def keybert_embedding(text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sbert_model = SentenceTransformer('distilbert-base-nli-mean-tokens', device=device)
    model = KeyBERT(model=sbert_model)
    keywords = model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=5)
    return keywords

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text embeddings using KeyBERT")
    parser.add_argument("text_file", type=str, help="Path to the text file to be embedded")
    
    args = parser.parse_args()
    
    # Read the text from the file
    with open(args.text_file, 'r') as f:
        text = f.read()
    
    keywords = keybert_embedding(text)
    print("KeyBERT Keywords: ", keywords)
