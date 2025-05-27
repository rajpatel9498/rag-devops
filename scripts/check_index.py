import pickle
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def inspect_faiss_index():
    """Inspect the FAISS index and its metadata."""
    try:
        # Load the embeddings model
        logger.info("Loading embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Load the FAISS index
        logger.info("Loading FAISS index...")
        index_path = Path("embeddings/index.faiss")
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
            
        index = faiss.read_index(str(index_path))
        
        # Load metadata
        logger.info("Loading metadata...")
        metadata_path = Path("embeddings/index.pkl")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
            
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        
        # Print FAISS index statistics
        logger.info("\n=== FAISS Index Statistics ===")
        logger.info(f"Number of vectors: {index.ntotal}")
        logger.info(f"Vector dimension: {index.d}")
        
        # Print metadata statistics
        logger.info("\n=== Metadata Statistics ===")
        logger.info(f"Metadata type: {type(metadata)}")
        
        if isinstance(metadata, tuple):
            docs, id_map = metadata
            logger.info(f"Tuple length: {len(metadata)}")
            logger.info(f"First element type: {type(docs)}")
            logger.info(f"Second element type: {type(id_map)}")
            
            # Access documents using _dict
            if hasattr(docs, '_dict'):
                num_docs = len(docs._dict)
                logger.info(f"Number of documents: {num_docs}")
                
                # Sample first 3 documents
                logger.info("\n=== Sample Documents ===")
                for i, (doc_id, doc) in enumerate(list(docs._dict.items())[:3]):
                    logger.info(f"\nDocument {i+1} (ID: {doc_id}):")
                    logger.info(f"Content: {doc.page_content[:200]}...")
                    if hasattr(doc, 'metadata'):
                        logger.info(f"Metadata: {doc.metadata}")
            else:
                logger.warning("Document store does not have _dict attribute")
                logger.info(f"Available attributes: {dir(docs)}")
        
        # Test similarity search
        logger.info("\n=== Testing Similarity Search ===")
        test_query = "How do I fix a pod stuck in pending state?"
        query_vector = embeddings.embed_query(test_query)
        
        # Try different thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6]
        for threshold in thresholds:
            logger.info(f"\nSearching with threshold {threshold}:")
            D, I = index.search(np.array([query_vector]), k=5)  # Get top 5 results
            
            # Filter results by threshold
            valid_results = [(dist, idx) for dist, idx in zip(D[0], I[0]) if dist <= threshold]
            
            if valid_results:
                logger.info(f"Found {len(valid_results)} results:")
                for dist, idx in valid_results:
                    if hasattr(docs, '_dict') and idx in docs._dict:
                        doc = docs._dict[idx]
                        logger.info(f"\nDistance: {dist:.4f}")
                        logger.info(f"Content: {doc.page_content[:200]}...")
                        if hasattr(doc, 'metadata'):
                            logger.info(f"Metadata: {doc.metadata}")
            else:
                logger.info(f"No results found with threshold {threshold}")
        
    except Exception as e:
        logger.error(f"Error inspecting index: {str(e)}")
        raise

if __name__ == "__main__":
    inspect_faiss_index() 