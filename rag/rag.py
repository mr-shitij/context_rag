import json
import os
from pprint import pprint
from typing import List, Dict, Any

import anthropic
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
        # self.pre_processor = PDFPreProcessor(raw_dir, processed_dir, group_size=group_size)
        # if not self.pre_processor.check_preprocessing():
        #     self.pre_processor.process()

        self.vector_db = ContextualVectorDB(
            collection_name=collection_name,
            voyage_api_key=voyage_api_key,
            anthropic_api_key=anthropic_api_key,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password
        )

        # if not self._prepare_documents():
        #     for hash_dir in os.listdir(processed_dir):
        #         json_path = os.path.join(processed_dir, hash_dir, "grouped_pages.json")
        #         if os.path.exists(json_path):
        #             print(f"Processing {hash_dir}...")
        #             with open(json_path, 'r', encoding='utf-8') as f:
        #                 json_data = json.load(f)
        #                 context_exists = any('context' in chunk for group in json_data for chunk in group['chunks'])
        #                 if context_exists:
        #                     print(f"Skipping {hash_dir} - context already exists")
        #                     continue
        #                 print(f"Processing {hash_dir}...")
        #                 self.vector_db.load_data(json_data, json_path, parallel_threads=2)

        self.re_ranker = ReRanker(anthropic_api_key)
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

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
        if len(os.listdir(processed_path)) == 0:
            all_ready = False
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
        reranked = self.re_ranker.rerank_candidates(query, candidates)
        print("Reranked Candidates: ")
        pprint(reranked)
        return reranked

    def query_llm(self, query: str) -> str:
        """
        Processes a query by first using an LLM call to decide whether retrieval augmentation is needed.
        If yes, it uses the search function to retrieve relevant documents, incorporates them as references,
        and then generates a final answer. Otherwise, it generates an answer directly.
        Returns the final answer as a string.
        """
        # Step 1: Decide whether to use RAG.
        use_rag = self._llm_decide_rag(query)
        print(f"LLM decision to use RAG: {use_rag}")

        if use_rag:
            # Step 2: Run the search function to retrieve relevant documents.
            search_results = self.search(query, k=10)
            # Step 3: Generate the answer using the retrieved references.
            final_answer = self._llm_generate_with_references(query, search_results)
        else:
            # Generate answer directly without RAG.
            final_answer = self._llm_generate_without_references(query)

        return final_answer

    def _llm_decide_rag(self, query: str) -> bool:
        """
        Uses an LLM to decide if the given query would benefit from retrieval augmentation.
        """
        prompts = {
            'document': f"<query>\n{query}\n</query>",
            'query': "Determine whether retrieval-augmented generation (RAG) is needed for this query. Respond by using only "
                     "'yes' or 'no'."
        }
        try:
            response = self.anthropic_client.messages.create(
                model=MODEL_NAME,
                max_tokens=10,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompts['document']},
                        {"type": "text", "text": prompts['query']},
                    ]
                }],
            )

            print(f"LLM decision to use RAG: {response}")

            decision = None
            for content in response.content:
                if content.type == "text":
                    decision = content.text  # This might be a string.
                    break

            return str(decision).strip().lower().startswith("yes")
        except Exception as e:
            print("Error during RAG decision:", e)
            return True  # Default to using RAG if an error occurs.

    def _llm_generate_with_references(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Uses the LLM to generate an answer to the query using the provided references.
        """
        prompts = {
            'document': f"<search_result>{search_results}</search_result>\n\n<query>\n{query}\n</query>\n\n",
            'query': "Generate a detailed answer of query using the provided search_result and include citations ie. the doc id and chunk id."
        }
        return self._call_llm(prompts)

    def _llm_generate_without_references(self, query: str) -> str:
        """
        Uses the LLM to generate an answer to the query without any external references.
        """
        prompts = {
            'document': f"<query>\n{query}\n</query>",
            'query': "Generate an answer to the query. make the things very friendly."
        }
        return self._call_llm(prompts)

    def _call_llm(self, prompts: Dict[str, str]) -> str:
        """
        Calls the LLM to process a prompt.
        """
        try:
            response = self.anthropic_client.messages.create(
                model=MODEL_NAME,
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompts['document']},
                        {"type": "text", "text": prompts['query']},
                    ]
                }],
            )
            for content in response.content:
                if content.type == "text":
                    return str(content.text)
            return "Error: No valid response received."
        except Exception as e:
            print("Error during LLM call:", e)
            return "Error: Unable to generate an answer."


# --------------------------- Example Usage ---------------------------
if __name__ == "__main__":
    rag_system = RAG(
        collection_name="pdf_embeddings",
        voyage_api_key="pa-QhwbHHG0NSWxFv1uw-0KReqcnG8_kjCT8K1OOj3sKf8",
        anthropic_api_key="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",
        neo4j_uri="neo4j+s://9fb25f55.databases.neo4j.io",
        neo4j_user="neo4j",
        neo4j_password="wbVkkp6WbC_fruL0qifiCL0eezQP9rpGvEHeoobCkBw",
        raw_dir="../DOCS/raw",
        processed_dir="../DOCS/processed",
    )

    query = "tell me about agreement of exim bank"
    final_answer = rag_system.query_llm(query)
    print("\nFinal Answer:")
    print(final_answer)
