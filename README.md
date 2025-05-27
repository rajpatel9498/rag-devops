# Kubernetes Issue Management using RAG

This project implements a Retrieval-Augmented Generation (RAG) system for managing and analyzing Kubernetes GitHub issues. It uses FAISS for efficient vector similarity search and integrates with LangChain and OpenAI GPT-4 for intelligent issue analysis and response generation.

## Features

- Fetch Kubernetes issues from GitHub using GraphQL API
- Preprocess and clean issue data
- Generate embeddings using sentence transformers
- Efficient similarity search using FAISS
- RAG-powered issue analysis and response generation
- FastAPI-based REST API for easy integration
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
│   └── langchain_rag.py        # Core RAG implementation
├── embeddings/              # Directory containing FAISS index
│   ├── index.faiss         # FAISS vector index (created by embed_issues.py)
│   └── index.pkl           # Document metadata (created by embed_issues.py)
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
   This will create two files in the `embeddings/` directory:
   - `index.faiss`: The FAISS vector index
   - `index.pkl`: Document metadata and mappings

4. Query the vector database:
   ```bash
   python scripts/query_vector_db.py
   ```
   This starts an interactive CLI where you can:
   - Enter questions about Kubernetes issues
   - Get top-k most relevant results with similarity scores
   - View issue content previews
   - Type 'exit' to quit

5. Evaluate the RAG system:
   ```bash
   python scripts/evaluate_rag.py
   ```

6. Validate RAG responses:
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

1. Make sure you've run the scripts in order:
   - `fetch_github_issues.py`
   - `preprocess_issues.py`
   - `embed_issues.py`
   before running `query_vector_db.py`

2. Check that the `embeddings/` directory contains:
   - `index.faiss`
   - `index.pkl`

3. Verify your environment variables in `.env`

4. Check the logs for detailed error messages

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 