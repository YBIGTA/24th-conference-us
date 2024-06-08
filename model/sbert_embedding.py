from sentence_transformers import SentenceTransformer

def sbert_embedding(text):
    device = "cpu"  # 마찬가지로 cpu로 바꿈
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)
    embedding = model.encode(text, device=device)
    return embedding