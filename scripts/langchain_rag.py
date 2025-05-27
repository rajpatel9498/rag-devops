from typing import List, Dict
import os
import logging
import time
from datetime import datetime
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import json
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check for required environment variables
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY environment variable is not set. "
        "Please set it in your .env file or environment."
    )

class KubernetesRAG:
    def __init__(self):
        try:
            logger.info("Initializing RAG system...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            self.llm = OpenAI(
                temperature=0,
                model_name="gpt-3.5-turbo-instruct",  # Using a more capable model
                max_tokens=1000
            )
            self.vector_store = None
            self.qa_chain = None
            self._load_vector_store()
            logger.info("✅ RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            raise

    def _load_vector_store(self):
        """Load the FAISS vector store and metadata"""
        try:
            logger.info("🔍 Loading FAISS index...")
            self.vector_store = FAISS.load_local(
                "embeddings",
                self.embeddings,
                index_name="index",
                allow_dangerous_deserialization=True
            )
            logger.info(f"✅ FAISS index loaded with {len(self.vector_store.index_to_docstore_id)} documents")
            
            # Enhanced prompt template with more context and structure
            prompt_template = """You are a Kubernetes expert assistant specialized in issue triage and troubleshooting. 
Your task is to provide accurate, helpful answers based on the provided context from GitHub issues.

Context Information:
{context}

User Question: {question}

Instructions:
1. Use ONLY the provided context to answer the question
2. If the context doesn't contain relevant information, say so
3. If you're unsure about any part, acknowledge the uncertainty
4. Include specific issue numbers or references when relevant
5. Format your response in a clear, structured way

Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template, 
                input_variables=["context", "question"]
            )
            
            logger.info("🔧 Configuring retriever...")
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={
                        "k": 3,  # Reduced to avoid context overflow
                        "fetch_k": 5,
                        "search_type": "similarity",
                    }
                ),
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": PROMPT,
                    "document_variable_name": "context",
                    "document_prompt": PromptTemplate(
                        template="Issue #{number}: {page_content_short}\nURL: {url}\n---\n",
                        input_variables=["number", "page_content_short", "url"]
                    )
                }
            )
            logger.info("✅ Retriever configured successfully")
            
        except FileNotFoundError as e:
            logger.error(f"Error: Required files not found. Have you run the preprocessing and embedding scripts?")
            logger.error(f"Expected files: embeddings/index.faiss and embeddings/index.pkl")
            raise
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise

    def query(self, question: str) -> Dict:
        """
        Query the RAG system with a question about Kubernetes issues
        
        Args:
            question (str): The question to ask
            
        Returns:
            Dict: Contains answer, source documents, and performance metrics
        """
        start_time = time.time()
        try:
            logger.info(f"Processing question: {question[:100]}...")
            
            # Get answer from QA chain
            result = self.qa_chain({"query": question})
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Get similarity scores from the retriever
            docs_with_scores = self.vector_store.similarity_search_with_score(question, k=5)
            
            # Prepare short content for prompt
            def doc_with_short_content(doc):
                doc.metadata = dict(doc.metadata) if doc.metadata else {}
                doc.metadata["page_content_short"] = doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else "")
                return doc
            # Patch source documents for prompt
            if "source_documents" in result:
                result["source_documents"] = [doc_with_short_content(doc) for doc in result["source_documents"]]
            
            # Format the response with enhanced metadata
            response = {
                "answer": result["result"],
                "sources": [
                    {
                        "issue_number": doc.metadata.get("number", "Unknown"),
                        "title": doc.metadata.get("title", "Unknown"),
                        "url": doc.metadata.get("url", "Unknown"),
                        "content": doc.page_content[:200] + "...",
                        "similarity_score": float(score)  # Convert score to float
                    }
                    for doc, score in docs_with_scores
                ],
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "num_sources": len(docs_with_scores),
                    "avg_similarity_score": sum(score for _, score in docs_with_scores) / len(docs_with_scores) if docs_with_scores else 0.0
                }
            }
            
            logger.info(f"Query processed in {processing_time:.2f} seconds")
            logger.info(f"Found {len(response['sources'])} relevant sources")
            
            return response
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return {
                "error": str(e),
                "answer": "I encountered an error while processing your question.",
                "sources": [],
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }

def main():
    """CLI interface for the RAG system"""
    rag = KubernetesRAG()
    
    print("🤖 Kubernetes Issue Assistant (Type 'exit' to quit)")
    print("=" * 50)
    
    while True:
        question = input("\n❓ Your question: ").strip()
        
        if question.lower() == "exit":
            break
            
        response = rag.query(question)
        
        print("\n📝 Answer:")
        print(response["answer"])
        
        print("\n🔍 Sources:")
        for i, source in enumerate(response["sources"], 1):
            print(f"\n{i}. Issue #{source['issue_number']}")
            print(f"   URL: {source['url']}")
            print(f"   Preview: {source['content']}")

if __name__ == "__main__":
    main() 