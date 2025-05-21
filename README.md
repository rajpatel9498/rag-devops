# Kubernetes Issue Management using RAG

This project implements a Retrieval-Augmented Generation (RAG) system for managing and analyzing Kubernetes GitHub issues. It uses FAISS for efficient vector similarity search and integrates with LangChain and OpenAI GPT-4 for intelligent issue analysis and response generation.

## Features

- Fetch Kubernetes issues from GitHub using GraphQL API
- Preprocess and clean issue data
- Generate embeddings using sentence transformers
- Efficient similarity search using FAISS
- RAG-powered issue analysis and response generation
- FastAPI-based REST API for easy integration

## Project Structure

```
rag-devops/
├── scripts/                  # Python scripts for each phase
│   ├── fetch_github_issues.py  # GitHub issue fetching
│   ├── preprocess_issues.py    # Data preprocessing
│   ├── embed_issues.py         # Embedding generation
│   └── query_vector_db.py      # Vector similarity search
├── .env                      # Environment variables (not tracked)
├── .gitignore               # Git ignore rules
├── requirements.txt         # Python dependencies
└── README.md               # This file
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

3. Generate embeddings:
   ```bash
   python scripts/embed_issues.py
   ```

4. Query the vector database:
   ```bash
   python scripts/query_vector_db.py
   ```

## Development

- The project uses Python 3.8+
- Dependencies are managed through `requirements.txt`
- Code style follows PEP 8 guidelines
- Each script is modular and well-documented

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 