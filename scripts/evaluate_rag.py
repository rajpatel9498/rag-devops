import csv
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Load FAISS vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
db = FAISS.load_local("embeddings", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# RAG chain setup
llm = ChatOpenAI()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Define test queries
test_queries = [
    "How can I filter Kubernetes replication controllers using labels?",
    "What is the difference between Deployment and StatefulSet in Kubernetes?",
    "How to use labels for filtering Kubernetes pods?",
    "How do I delete a stuck Kubernetes namespace?",
    "How to create a ConfigMap from a file?",
    "Explain rolling update strategy in Kubernetes.",
    "How do I access logs from a crashed pod?",
    "How can I use taints and tolerations?",
    "How to mount a volume in a pod?",
    "What are common Kubernetes troubleshooting commands?"
]

# Prepare CSV output
output_file = "rag_evaluation_results.csv"
fieldnames = ["Query", "GeneratedAnswer", "ResponseTimeSeconds", "ManualRelevanceScore"]

with open(output_file, "w", newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for query in test_queries:
        start_time = time.time()
        result = qa_chain({"query": query})
        end_time = time.time()
        response_time = round(end_time - start_time, 2)

        writer.writerow({
            "Query": query,
            "GeneratedAnswer": result["result"],
            "ResponseTimeSeconds": response_time,
            "ManualRelevanceScore": ""  # Placeholder for human annotation
        })

print(f"Evaluation completed. Results saved to {output_file}")
