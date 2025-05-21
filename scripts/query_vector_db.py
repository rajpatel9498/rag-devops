# scripts/query_vector_db.py

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# File paths
vector_path = "embeddings/faiss_index.bin"
metadata_path = "embeddings/metadata.pkl"

# Load model and vector DB
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
index = faiss.read_index(vector_path)

with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)

# Search function
def search(query, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    
    print(f"\n🔍 Top {top_k} results for query: '{query}'")
    print("-" * 60)
    for i, idx in enumerate(indices[0]):
        match = metadata[idx]
        print(f"#{i+1}: Issue #{match['number']}")
        print(f"     URL: {match['url']}")
        print(f"     Score: {distances[0][i]:.2f}")
        print()

# Interactive CLI
if __name__ == "__main__":
    while True:
        query = input("🧠 Enter a GitHub issue question (or 'exit'): ").strip()
        if query.lower() == "exit":
            break
        search(query)
