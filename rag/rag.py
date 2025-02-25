import json
from pprint import pprint
from typing import List, Dict, Any

import matplotlib
from pathlib import Path

from pre_processor import PDFPreProcessor
from reranker import ReRanker
from vector_db import ContextualVectorDB

matplotlib.use("Agg")  # for saving static images without a display

MODEL_NAME = "claude-3-haiku-20240307"


# --------------------------- RAG Class ---------------------------
class RAG:
    def __init__(self, collection_name: str, raw_dir: str, processed_dir: str,
                 voyage_api_key=None, anthropic_api_key=None, neo4j_uri=None,
                 neo4j_user=None, neo4j_password=None, group_size=50):
        """
        Initialize the RAG system with a vector database instance and a re-ranker.
        """
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.pre_processor = PDFPreProcessor(raw_dir, processed_dir, group_size=group_size)
        if not self.pre_processor.check_preprocessing():
            self.pre_processor.process_all()

        self.vector_db = ContextualVectorDB(
            collection_name=collection_name,
            voyage_api_key=voyage_api_key,
            anthropic_api_key=anthropic_api_key,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password
        )
        self.re_ranker = ReRanker(anthropic_api_key)

    def _prepare_documents(self) -> bool:
        """
        Checks if the processed documents are ready. For each hash directory in self.processed_dir,
        verifies that a 'grouped_pages.json' file exists and that each chunk in the file has a non-empty 'context' field.

        Returns:
            True if all documents are ready; otherwise, False.
        """
        processed_path = Path(self.processed_dir)
        if not processed_path.exists():
            print(f"Processed directory {self.processed_dir} does not exist. Please run the pre-processing step.")
            return False

        all_ready = True
        for hash_dir in processed_path.iterdir():
            if hash_dir.is_dir():
                json_file = hash_dir / "grouped_pages.json"
                if not json_file.exists():
                    print(f"Missing JSON file in {hash_dir}. Please run the pre-processing step for this directory.")
                    all_ready = False
                else:
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        # Check each group and its chunks for a non-empty 'context'
                        for group in data:
                            for chunk in group.get("chunks", []):
                                if not chunk.get("context"):
                                    print(f"Chunk {chunk.get('id')} in {hash_dir} is missing context.")
                                    all_ready = False
                                    break  # Exit inner loop if any chunk is missing context.
                            if not all_ready:
                                break  # Exit group loop if processing is required.
                    except Exception as e:
                        print(f"Error reading {json_file}: {e}")
                        all_ready = False
        if all_ready:
            print("Pre-processing check complete. All documents appear to be ready.")
        else:
            print("Some documents are missing context. Pre-processing is required.")
        return all_ready

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
        neo4j_password="hY2rdVwzBb0FDh8nABwsXYwGsjiINdEzY0KINb5h1jI",
        raw_dir="../DOCS/raw",
        processed_dir="../DOCS/processed",
    )

    final_results = rag_system.search("road optimization", k=10)
    print("\nMerged and Re-ranked Results:")
    for res in final_results:
        pprint(res)
