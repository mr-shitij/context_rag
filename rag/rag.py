import json
from pprint import pprint
from typing import List, Dict, Any

import matplotlib

from reranker import ReRanker
from vector_db import ContextualVectorDB

matplotlib.use("Agg")  # for saving static images without a display

MODEL_NAME = "claude-3-haiku-20240307"


# --------------------------- RAG Class ---------------------------
class RAG:
    def __init__(self, collection_name: str, voyage_api_key=None, anthropic_api_key=None, neo4j_uri=None,
                 neo4j_user=None, neo4j_password=None):
        """
        Initialize the RAG system with a vector database instance and a re-ranker.
        """
        self.vector_db = ContextualVectorDB(
            collection_name=collection_name,
            voyage_api_key=voyage_api_key,
            anthropic_api_key=anthropic_api_key,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password
        )
        self.re_ranker = ReRanker(anthropic_api_key)

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves candidate chunks via the vector_graph_search method, flattens the candidate structure,
        and uses the LLM-based re-ranker to re-order them.
        Returns the top 10 re-ranked candidate dictionaries.
        """
        candidates = self.vector_db.vector_graph_search(query, k=k)
        pprint(candidates)
        print("Candidates are printed")
        reranked = self.re_ranker.rerank_candidates(query, candidates)
        return reranked


# --------------------------- Example Usage ---------------------------
if __name__ == "__main__":
    rag_system = RAG(
        collection_name="pdf_embeddings",
        voyage_api_key="pa-QhwbHHG0NSWxFv1uw-0KReqcnG8_kjCT8K1OOj3sKf8",
        anthropic_api_key="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",
        neo4j_uri="neo4j+s://e9882b6e.databases.neo4j.io",
        neo4j_user="neo4j",
        neo4j_password="hY2rdVwzBb0FDh8nABwsXYwGsjiINdEzY0KINb5h1jI")

    final_results = rag_system.search("road optimization", k=10)
    print("\nMerged and Re-ranked Results:")
    for res in final_results:
        pprint(res)
