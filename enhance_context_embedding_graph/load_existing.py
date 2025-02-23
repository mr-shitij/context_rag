import json
import os
from typing import List, Dict, Any

from tqdm import tqdm

from enhance_context_embedding_graph.contextExtractor import ContextualVectorDB


# Usage in main:
if __name__ == "__main__":
    db = ContextualVectorDB("pdf_embeddings")

    # Process JSONs from hash directories
    base_dir = "../DOCS"
    processed_dir = os.path.join(base_dir, "processed")

    for hash_dir in os.listdir(processed_dir):
        json_path = os.path.join(processed_dir, hash_dir, "grouped_pages.json")
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

                # Check if context exists
                context_exists = any(
                    'context' in chunk
                    for group in json_data
                    for chunk in group['chunks']
                )

                if context_exists:
                    print(f"Loading existing contexts from {hash_dir}")
                    db.load_existing_json(json_data)
                else:
                    print(f"Processing new file {hash_dir}...")
                    db.load_data(json_data, json_path, parallel_threads=4)
