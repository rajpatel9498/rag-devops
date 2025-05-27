# scripts/embed_issues.py

import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Paths
input_path = "data/k8s_issues_preprocessed.json"
embeddings_dir = "embeddings"

# Load data
def load_issues():
    if not os.path.exists(input_path):
        raise FileNotFoundError("Run preprocessing first.")
    with open(input_path, "r") as f:
        return json.load(f)

def main():
    os.makedirs(embeddings_dir, exist_ok=True)
    issues = load_issues()

    # Prepare texts and metadata
    texts = [issue["text"] for issue in issues]
    metadatas = [{"number": str(issue["number"]), "url": issue["url"]} for issue in issues]

    # Initialize embeddings
    print("🔍 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    print("🔢 Generating embeddings and creating FAISS index...")
    # Create FAISS index using LangChain's implementation
    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    # Save the index
    print("💾 Saving FAISS index...")
    vector_store.save_local(embeddings_dir)
    print(f"✅ Saved FAISS index to {embeddings_dir}/")

if __name__ == "__main__":
    main()