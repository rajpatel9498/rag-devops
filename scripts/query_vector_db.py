# scripts/query_vector_db.py

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File paths
vector_path = "embeddings/index.faiss"
metadata_path = "embeddings/index.pkl"

try:
    # Load model and vector DB
    logger.info("Loading sentence transformer model...")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    
    logger.info("Loading FAISS index...")
    index = faiss.read_index(vector_path)
    
    logger.info("Loading metadata...")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
        
    # Handle metadata structure (could be tuple of (docs, id_map) from LangChain)
    if isinstance(metadata, tuple) and len(metadata) == 2:
        docs, id_map = metadata
        if hasattr(docs, '_dict'):
            # Convert to list of documents for easier access
            doc_list = list(docs._dict.values())
            logger.info(f"Loaded {len(doc_list)} documents")
        else:
            raise ValueError("Unexpected metadata structure: docs object has no _dict attribute")
    else:
        raise ValueError("Unexpected metadata structure: expected tuple of (docs, id_map)")

except FileNotFoundError as e:
    logger.error(f"Error: Could not find required files. Make sure you've run embed_issues.py first.")
    logger.error(f"Expected files: {vector_path} and {metadata_path}")
    raise
except Exception as e:
    logger.error(f"Error loading index or metadata: {str(e)}")
    raise

# Search function
def search(query, top_k=5):
    try:
        query_embedding = model.encode([query])
        distances, indices = index.search(np.array(query_embedding), top_k)
        
        print(f"\n🔍 Top {top_k} results for query: '{query}'")
        print("-" * 60)
        
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx < len(doc_list):
                doc = doc_list[idx]
                print(f"#{i+1}: Issue #{doc.metadata.get('number', 'Unknown')}")
                print(f"     URL: {doc.metadata.get('url', 'Unknown')}")
                print(f"     Score: {distance:.2f}")
                print(f"     Content: {doc.page_content[:200]}...")
                print()
            else:
                logger.warning(f"Index {idx} out of range for document list")
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        raise

# Interactive CLI
if __name__ == "__main__":
    try:
        while True:
            query = input("🧠 Enter a GitHub issue question (or 'exit'): ").strip()
            if query.lower() == "exit":
                break
            search(query)
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
        raise
