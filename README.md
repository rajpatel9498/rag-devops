# Kubernetes Issue Management using RAG

This project implements a Retrieval-Augmented Generation (RAG) system for managing and analyzing Kubernetes GitHub issues. It uses FAISS for efficient vector similarity search and integrates with LangChain and OpenAI GPT-4 for intelligent issue analysis and response generation. The system includes both a FastAPI backend and a Streamlit web interface for easy interaction.

## Features

- Fetch Kubernetes issues from GitHub using GraphQL API
- Preprocess and clean issue data
- Generate embeddings using sentence transformers
- Efficient similarity search using FAISS
- RAG-powered issue analysis and response generation
- FastAPI-based REST API for programmatic access
- Interactive Streamlit web interface for user-friendly interaction
- Real-time question answering with source attribution
- Chat history and performance metrics
- Interactive CLI for direct vector database queries

## Project Structure

```
rag-devops/
├── scripts/                  # Python scripts for each phase
│   ├── fetch_github_issues.py  # GitHub issue fetching
│   ├── preprocess_issues.py    # Data preprocessing
│   ├── embed_issues.py         # Embedding generation
│   ├── query_vector_db.py      # Interactive vector similarity search
│   ├── evaluate_rag.py         # RAG system evaluation
│   ├── validate_rag.py         # Validation of RAG responses
│   ├── langchain_rag.py        # Core RAG implementation
│   ├── fastapi_rag.py         # FastAPI backend server
│   └── streamlit_app.py       # Streamlit web interface
├── embeddings/              # Directory containing FAISS index
│   ├── index.faiss         # FAISS vector index
│   └── index.pkl           # Document metadata
├── data/                   # Data directory
│   ├── k8s_issues.json     # Raw GitHub issues
│   └── k8s_issues_preprocessed.json  # Preprocessed issues
├── .env                    # Environment variables (not tracked)
├── .gitignore             # Git ignore rules
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/rag-devops.git
   cd rag-devops
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys:
   ```
   GITHUB_TOKEN=your_github_token
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

### Data Processing Pipeline

1. Fetch Kubernetes issues:
   ```bash
   python scripts/fetch_github_issues.py
   ```

2. Preprocess the issues:
   ```bash
   python scripts/preprocess_issues.py
   ```

3. Generate embeddings and create FAISS index:
   ```bash
   python scripts/embed_issues.py
   ```

### Web Interface

1. Start the FastAPI backend server:
   ```bash
   uvicorn scripts.fastapi_rag:app --reload
   ```

2. In a new terminal, start the Streamlit web interface:
   ```bash
   streamlit run scripts/streamlit_app.py
   ```

3. Open your browser and navigate to `http://localhost:8501`

The web interface provides:
- A clean, modern UI for asking questions
- Real-time responses with source attribution
- Chat history of previous questions
- Performance metrics
- Links to source GitHub issues

### API Access

The FastAPI backend provides a REST API endpoint at `http://localhost:8000/query`:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/query' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "How do I filter Kubernetes pods by labels?"
  }'
```

### CLI Tools

1. Query the vector database directly:
   ```bash
   python scripts/query_vector_db.py
   ```

2. Evaluate the RAG system:
   ```bash
   python scripts/evaluate_rag.py
   ```

3. Validate RAG responses:
   ```bash
   python scripts/validate_rag.py
   ```

## Development

- The project uses Python 3.8+
- Dependencies are managed through `requirements.txt`
- Code style follows PEP 8 guidelines
- Each script is modular and well-documented
- Logging is implemented for better debugging
- Error handling is implemented throughout the codebase

## Troubleshooting

If you encounter issues:

1. Make sure you've run the data processing pipeline in order:
   - `fetch_github_issues.py`
   - `preprocess_issues.py`
   - `embed_issues.py`

2. Check that the `embeddings/` directory contains:
   - `index.faiss`
   - `index.pkl`

3. Verify your environment variables in `.env`

4. For web interface issues:
   - Ensure both FastAPI and Streamlit servers are running
   - Check the browser console for any errors
   - Verify the backend URL in `streamlit_app.py`

5. Check the logs for detailed error messages:
   - FastAPI logs in the terminal
   - Streamlit logs in the terminal
   - RAG system logs in `rag_system.log`

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 