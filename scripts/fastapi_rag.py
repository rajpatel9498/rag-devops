from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import time
from datetime import datetime
import uvicorn
from scripts.langchain_rag import KubernetesRAG  # Modify the import path
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="Kubernetes Issue RAG API",
    description="API for querying Kubernetes issues using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Initialize RAG system
rag_system = None

class QueryRequest(BaseModel):
    question: str

class Source(BaseModel):
    issue_number: str
    url: str
    content: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    processing_time: float
    timestamp: str

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    global rag_system
    try:
        rag_system = KubernetesRAG()
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Kubernetes Issue RAG API",
        "version": "1.0.0"
    }

@app.post("/query", response_model=QueryResponse)
@limiter.limit("5/minute")
async def query(request: Request, query_request: QueryRequest):
    """
    Query the RAG system with a question about Kubernetes issues
    
    Args:
        request (Request): The FastAPI request object
        query_request (QueryRequest): The query request containing the question
        
    Returns:
        QueryResponse: The response containing the answer and sources
    """
    if not rag_system:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not initialized"
        )
    
    start_time = time.time()
    
    try:
        # Get response from RAG system
        response = rag_system.query(query_request.question)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Format response
        return QueryResponse(
            answer=response["answer"],
            sources=[
                Source(
                    issue_number=source["issue_number"],
                    url=source["url"],
                    content=source["content"]
                )
                for source in response["sources"]
            ],
            processing_time=processing_time,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

def start():
    """Start the FastAPI server"""
    uvicorn.run(
        "fastapi_rag:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    start() 