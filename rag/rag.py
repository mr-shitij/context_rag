# rag_system.py

from typing import List, Dict, Any
from reranker import ReRanker  # Ensure that ReRanker is in reranker.py


class RAGSystem:
    def __init__(self, vector_db, reranker: ReRanker):
        """
        Initialize the RAG system with a vector database instance and a re-ranker instance.
        """
        self.vector_db = vector_db
        self.reranker = reranker

    def search(self, query: str, vector_k: int = 10, graph_k: int = 10) -> List[Dict[str, Any]]:
        """
        Performs the full RAG search:
          1. Vector search via Milvus.
          2. Graph search via Neo4j.
          3. Re-rank vector candidates (top 5).
          4. Re-rank graph candidates (top 5).
          5. Merge and re-rank the final candidates.
        Returns a list of the top candidate chunks.
        """
        # Perform vector search.
        vector_results = self.vector_db.search(query, k=vector_k)
        print("Vector search results:")
        for res in vector_results:
            print(res)

        # Perform graph search.
        graph_results = self.vector_db.search_neo4j(query)
        print("Graph search results:")
        for res in graph_results:
            print(res)

        # Re-rank each set.
        top_vector = self.reranker.rank_vector_candidates(query, vector_results)
        top_graph = self.reranker.rank_graph_candidates(query, graph_results)

        print("\nTop Vector Candidates (Re-ranked):")
        for cand in top_vector:
            print(cand)
        print("\nTop Graph Candidates (Re-ranked):")
        for cand in top_graph:
            print(cand)

        # Merge and final re-rank.
        final_candidates = self.reranker.merge_and_rerank(query, vector_results, graph_results)
        print("\nFinal Top Candidates:")
        for cand in final_candidates:
            print(cand)

        return final_candidates


# Example usage:
if __name__ == "__main__":
    from vector_db import ContextualVectorDB  # Ensure correct import based on your project structure

    # Instantiate your vector database.
    db = ContextualVectorDB(collection_name="pdf_embeddings")
    # Instantiate the re-ranker.
    from reranker import ReRanker

    reranker = ReRanker()

    # Create the RAG system.
    rag_system = RAGSystem(vector_db=db, reranker=reranker)

    # Run a query.
    query = "What are the key challenges in road optimization using MCDM?"
    final_results = rag_system.search(query, vector_k=10, graph_k=10)
