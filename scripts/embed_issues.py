# scripts/embed_issues.py

import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Paths
input_path = "data/k8s_issues_preprocessed.json"
vector_path = "embeddings/faiss_index.bin"
metadata_path = "embeddings/metadata.pkl"

# Load data
def load_issues():
    if not os.path.exists(input_path):
        raise FileNotFoundError("Run preprocessing first.")
    with open(input_path, "r") as f:
        return json.load(f)

def main():
    os.makedirs("embeddings", exist_ok=True)
    issues = load_issues()

    texts = [issue["text"] for issue in issues]
    metadata = [{"number": issue["number"], "url": issue["url"]} for issue in issues]

    # Load model
    print("🔍 Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    print("🔢 Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # Save metadata
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    faiss.write_index(index, vector_path)
    print(f"✅ Saved FAISS index to {vector_path}")
    print(f"📎 Saved metadata to {metadata_path}")

if __name__ == "__main__":
    main()