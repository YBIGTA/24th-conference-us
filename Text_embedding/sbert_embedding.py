# sbert_embedding.py
from sentence_transformers import SentenceTransformer
import torch
import argparse

def sbert_embedding(text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)
    embedding = model.encode(text, device=device)
    return embedding

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text embeddings using SBERT")
    parser.add_argument("text_file", type=str, help="Path to the text file to be embedded")
    
    args = parser.parse_args()
    
    # Read the text from the file
    with open(args.text_file, 'r') as f:
        text = f.read()
    
    embedding = sbert_embedding(text)
    print("SBERT Embedding: ", embedding)
