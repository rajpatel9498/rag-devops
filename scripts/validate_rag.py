import csv
import json
import time
from datetime import datetime
from langchain_rag import KubernetesRAG

# Define test queries
TEST_QUERIES = [
    "How do I restart a Kubernetes pod?",
    "What is the difference between Deployment and StatefulSet in Kubernetes?",
    "How to use labels for filtering Kubernetes pods?",
    "How do I delete a stuck Kubernetes namespace?",
    "How to create a ConfigMap from a file?",
    "Explain rolling update strategy in Kubernetes.",
    "How do I access logs from a crashed pod?",
    "How can I use taints and tolerations?",
    "How to mount a volume in a pod?",
    "How can Kubernetes handle retrieving deleted resources without relying on object names, especially when those names might get reused immediately?"
]

CSV_FIELDS = [
    "Query",
    "GeneratedAnswer",
    "ResponseTimeSeconds",
    "ManualRelevanceScore",
    "TopDocuments"
]


def summarize_doc(doc):
    # Summarize document for output (first 200 chars)
    return {
        "issue_number": doc.get("issue_number", ""),
        "title": doc.get("title", ""),
        "url": doc.get("url", ""),
        "similarity_score": doc.get("similarity_score", ""),
        "content": doc.get("content", "")[:200] + ("..." if len(doc.get("content", "")) > 200 else "")
    }


def run_validation():
    rag = KubernetesRAG()
    results = []
    total_time = 0.0

    for query in TEST_QUERIES:
        start = time.time()
        response = rag.query(query)
        end = time.time()
        response_time = round(end - start, 2)
        total_time += response_time

        # Get top 3 docs (if available)
        top_docs = [summarize_doc(doc) for doc in response.get("sources", [])[:3]]

        results.append({
            "Query": query,
            "GeneratedAnswer": response.get("answer", ""),
            "ResponseTimeSeconds": response_time,
            "ManualRelevanceScore": "",  # Placeholder for human annotation
            "TopDocuments": top_docs
        })

    avg_time = total_time / len(TEST_QUERIES)
    print(f"\nAverage response time: {avg_time:.2f} seconds")

    # Save to CSV
    csv_file = "rag_validation_results.csv"
    with open(csv_file, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in results:
            # Serialize TopDocuments as JSON string for CSV
            row_copy = row.copy()
            row_copy["TopDocuments"] = json.dumps(row_copy["TopDocuments"], ensure_ascii=False)
            writer.writerow(row_copy)
    print(f"Results saved to {csv_file}")

    # Save to JSON
    json_file = "rag_validation_results.json"
    with open(json_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {json_file}")


def cli():
    """Simple CLI to run batch validation"""
    print("Running RAG validation for test queries...")
    run_validation()


if __name__ == "__main__":
    cli()

# ---
# How to calculate precision, recall, MTTR (if ground truth is available):
#
# - Precision: For each query, count how many retrieved docs are truly relevant (per ground truth),
#   divide by total retrieved docs. Average over all queries.
# - Recall: For each query, count how many relevant docs were retrieved, divide by total relevant docs in ground truth.
# - MTTR (Mean Time To Resolution): If you have timestamps for when issues were created and resolved,
#   compute the average time between creation and resolution for resolved issues.
#
# You can add ground truth labels to the CSV/JSON and write a script to compute these metrics later. 