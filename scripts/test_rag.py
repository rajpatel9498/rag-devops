import unittest
import json
import time
from datetime import datetime
from typing import List, Dict
from langchain_rag import KubernetesRAG

class TestKubernetesRAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize the RAG system once for all tests"""
        cls.rag = KubernetesRAG()
        cls.test_questions = [
            {
                "question": "What are common issues with Kubernetes pod scheduling?",
                "expected_keywords": ["scheduling", "pod", "node", "resource"],
                "complexity": "basic"
            },
            {
                "question": "How do I troubleshoot a pod that is stuck in pending state?",
                "expected_keywords": ["pending", "troubleshoot", "describe", "events"],
                "complexity": "intermediate"
            },
            {
                "question": "What are the best practices for managing Kubernetes secrets in a production environment?",
                "expected_keywords": ["secrets", "security", "production", "management"],
                "complexity": "advanced"
            }
        ]

    def test_basic_query(self):
        """Test basic query functionality"""
        question = self.test_questions[0]["question"]
        response = self.rag.query(question)
        
        # Check response structure
        self.assertIn("answer", response)
        self.assertIn("sources", response)
        self.assertIn("processing_time", response)
        self.assertIn("metrics", response)
        
        # Check answer quality
        self.assertGreater(len(response["answer"]), 50)
        self.assertGreater(len(response["sources"]), 0)
        
        # Check processing time
        self.assertLess(response["processing_time"], 10.0)  # Should process within 10 seconds

    def test_source_relevance(self):
        """Test if retrieved sources are relevant to the question"""
        for test_case in self.test_questions:
            response = self.rag.query(test_case["question"])
            
            # Check if we got any sources
            self.assertGreater(len(response["sources"]), 0, 
                             f"No sources found for question: {test_case['question']}")
            
            # Check source metadata
            for source in response["sources"]:
                self.assertIn("issue_number", source)
                self.assertIn("url", source)
                self.assertIn("content", source)
                self.assertIn("similarity_score", source)
                
                # Check if similarity score is reasonable
                self.assertGreaterEqual(source["similarity_score"], 0.0)
                self.assertLessEqual(source["similarity_score"], 1.0)

    def test_answer_quality(self):
        """Test the quality of generated answers"""
        for test_case in self.test_questions:
            response = self.rag.query(test_case["question"])
            
            # Check if answer contains expected keywords
            answer_lower = response["answer"].lower()
            for keyword in test_case["expected_keywords"]:
                self.assertIn(
                    keyword.lower(),
                    answer_lower,
                    f"Expected keyword '{keyword}' not found in answer for question: {test_case['question']}"
                )

    def test_error_handling(self):
        """Test system's handling of invalid queries"""
        invalid_questions = [
            "",  # Empty question
            "   ",  # Whitespace only
            "a" * 1000,  # Very long question
            "What is the meaning of life?",  # Unrelated question
        ]
        
        for question in invalid_questions:
            response = self.rag.query(question)
            self.assertIn("answer", response)
            # System should handle invalid queries gracefully
            self.assertNotIn("error", response)

    def test_performance_metrics(self):
        """Test if performance metrics are being collected correctly"""
        response = self.rag.query(self.test_questions[0]["question"])
        
        # Check metrics structure
        metrics = response["metrics"]
        self.assertIn("num_sources", metrics)
        self.assertIn("avg_similarity_score", metrics)
        
        # Check metric values
        self.assertGreaterEqual(metrics["num_sources"], 0)
        self.assertGreaterEqual(metrics["avg_similarity_score"], 0.0)
        self.assertLessEqual(metrics["avg_similarity_score"], 1.0)

def run_performance_test():
    """Run a performance test with multiple queries"""
    rag = KubernetesRAG()
    results = []
    
    test_questions = [
        "What are common issues with Kubernetes pod scheduling?",
        "How do I fix a pod that is stuck in pending state?",
        "What are the best practices for managing Kubernetes secrets?",
        "How do I troubleshoot node not ready issues?",
        "What are the recommended resource limits for pods?"
    ]
    
    print("\n🔍 Running Performance Test")
    print("=" * 50)
    
    for question in test_questions:
        start_time = time.time()
        response = rag.query(question)
        end_time = time.time()
        
        result = {
            "question": question,
            "processing_time": response["processing_time"],
            "num_sources": len(response["sources"]),
            "avg_similarity": response["metrics"]["avg_similarity_score"],
            "answer_length": len(response["answer"])
        }
        results.append(result)
        
        print(f"\nQuestion: {question[:100]}...")
        print(f"Processing Time: {result['processing_time']:.2f}s")
        print(f"Sources Found: {result['num_sources']}")
        print(f"Average Similarity: {result['avg_similarity']:.2f}")
        print(f"Answer Length: {result['answer_length']} chars")
    
    # Calculate and print summary statistics
    avg_processing_time = sum(r["processing_time"] for r in results) / len(results)
    avg_sources = sum(r["num_sources"] for r in results) / len(results)
    avg_similarity = sum(r["avg_similarity"] for r in results) / len(results)
    
    print("\n📊 Performance Summary")
    print("=" * 50)
    print(f"Average Processing Time: {avg_processing_time:.2f}s")
    print(f"Average Sources per Query: {avg_sources:.1f}")
    print(f"Average Similarity Score: {avg_similarity:.2f}")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"test_results_{timestamp}.json", "w") as f:
        json.dump({
            "timestamp": timestamp,
            "summary": {
                "avg_processing_time": avg_processing_time,
                "avg_sources": avg_sources,
                "avg_similarity": avg_similarity
            },
            "detailed_results": results
        }, f, indent=2)
    
    print(f"\n✅ Results saved to test_results_{timestamp}.json")

if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # Run performance test
    run_performance_test() 