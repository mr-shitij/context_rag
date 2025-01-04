import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import anthropic
import voyageai
from pymilvus import MilvusClient
from pymilvus import Collection, connections
from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus.orm import utility
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)


class ContextualVectorDB:
    def __init__(self, collection_name: str, voyage_api_key=None, anthropic_api_key=None):
        """Initialize ContextualVectorDB with API clients and collection setup."""
        self.voyage_client = voyageai.Client(api_key=voyage_api_key or os.getenv("VOYAGE_API_KEY"))
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"))

        # Initialize Milvus client with proper URI
        self.collection_name = collection_name
        self.dimension = 1024
        self._setup_collection()

        self.token_counts = {'input': 0, 'output': 0, 'cache_read': 0, 'cache_creation': 0}
        self.token_lock = threading.Lock()

        self.requests_per_minute = 45
        self.last_request_time = 0
        self.request_lock = threading.Lock()

    def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limits."""
        with self.request_lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            if elapsed < 60 / self.requests_per_minute:
                time.sleep((60 / self.requests_per_minute) - elapsed)
            self.last_request_time = time.time()

    def _setup_collection(self):
        """Setup or get existing Milvus collection."""
        # Connect to Milvus
        connections.connect(host="localhost", port="19530")

        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            return

        # Define schema fields
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=False,
                description="primary id"
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dimension,
                description="embedding vector"
            ),
            FieldSchema(
                name="group_id",
                dtype=DataType.INT64,
                description="group identifier"
            ),
            FieldSchema(
                name="chunk_id",
                dtype=DataType.INT64,
                description="chunk identifier"
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="chunk content"
            ),
            FieldSchema(
                name="context",
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="chunk context"
            )
        ]

        # Create schema
        schema = CollectionSchema(
            fields=fields,
            description="PDF chunks with context",
            enable_dynamic_field=True
        )

        # Create collection
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            using='default',
        )

        # Create index for vector similarity search
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": 1024}
        }
        self.collection.create_index(
            field_name="vector",
            index_params=index_params
        )

    @retry(
        retry=retry_if_exception_type(anthropic.RateLimitError),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3)
    )
    def generate_context(self, group_text: str, chunk_text: str) -> tuple[str, Any]:
        """Generate context for a chunk using Claude."""
        prompts = {
            'document': "<document>\n{doc_content}\n</document>",
            'chunk': """Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document.
Answer only with the succinct context and nothing else."""
        }

        try:
            self._wait_for_rate_limit()
            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4096,
                temperature=0.0,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompts['document'].format(doc_content=group_text),
                            "cache_control": {"type": "ephemeral"}
                        },
                        {
                            "type": "text",
                            "text": prompts['chunk'].format(chunk_content=chunk_text)
                        }
                    ]
                }],
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
            )
            return response.content[0].text, response.usage
        except anthropic.RateLimitError as e:
            print(f"Rate limit hit, retrying with exponential backoff: {e}")
            raise

    def _process_chunk(self, group: Dict[str, Any], chunk: Dict[str, Any],
                       json_path: str, json_lock: threading.Lock) -> Dict[str, Any]:
        """Process a single chunk with context generation and embedding."""
        try:
            context, usage = self.generate_context(group['text'], chunk['text'])
            with self.token_lock:
                for key, value in usage.__dict__.items():
                    if key in self.token_counts:
                        self.token_counts[key] += value

            combined_text = f"{chunk['text']}\n\n{context}"
            embedding_result = self.voyage_client.embed([combined_text], model="voyage-2")

            # Update JSON file
            self._update_json_file(json_path, json_lock, group['id'], chunk['id'], context)

            return {
                "id": chunk['id'],
                "vector": embedding_result.embeddings[0],
                "group_id": group['id'],
                "chunk_id": chunk['id'],
                "content": chunk['text'],
                "context": context
            }
        except Exception as e:
            print(f"Error processing chunk {chunk['id']} in group {group['id']}: {e}")
            return None

    def _update_json_file(self, json_path: str, json_lock: threading.Lock,
                          group_id: int, chunk_id: int, context: str):
        """Update JSON file with new context."""
        with json_lock:
            with open(json_path, 'r', encoding='utf-8') as f:
                current_json = json.load(f)

            for group in current_json:
                if group['id'] == group_id:
                    for chunk in group['chunks']:
                        if chunk['id'] == chunk_id:
                            chunk['context'] = context
                            break
                    break

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(current_json, f, indent=2, ensure_ascii=False)

    def load_data(self, json_data: List[Dict[str, Any]], json_path: str, parallel_threads: int = 4):
        """Process and load data into Milvus collection."""
        total_chunks = sum(len(group['chunks']) for group in json_data)
        json_lock = threading.Lock()
        entities = []

        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            futures = []
            for group in json_data:
                for chunk in group['chunks']:
                    futures.append(executor.submit(
                        self._process_chunk, group, chunk, json_path, json_lock
                    ))

            for future in tqdm(as_completed(futures), total=total_chunks):
                result = future.result()
                if result:
                    entities.append(result)
                    if len(entities) >= 1000:
                        self._insert_batch(entities)
                        entities = []

        if entities:
            self._insert_batch(entities)

    def _insert_batch(self, entities: List[Dict[str, Any]]):
        """Insert batch of entities into Milvus."""
        try:
            batch = []
            # Format data according to schema
            for e in entities:
                data = {
                    "id": int(e["id"]),
                    "vector": e["vector"],
                    "group_id": int(e["group_id"]),
                    "chunk_id": int(e["chunk_id"]),
                    "content": str(e["content"]),
                    "context": str(e["context"])
                }
                batch.append(data)

            # Insert data
            self.collection.load()
            self.collection.insert(batch)
            print(f"Inserted {len(entities)} entities")
        except Exception as e:
            print(f"Error in batch insert: {e}")
            print(f"First entity structure: {entities[0]}")  # Debug info
            raise

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        """Perform similarity search in the collection."""
        # Generate query vector
        query_vector = self.voyage_client.embed([query], model="voyage-2").embeddings[0]

        # Load collection
        self.collection.load()

        try:
            # Perform search
            results = self.collection.search(
                data=[query_vector],
                anns_field="vector",
                param={"metric_type": "IP", "params": {"nprobe": 10}},
                limit=k,
                output_fields=["group_id", "chunk_id", "content", "context"]
            )

            # Format results
            return [{
                "metadata": {
                    "group_id": hit.entity.get('group_id'),
                    "chunk_id": hit.entity.get('chunk_id'),
                    "original_content": hit.entity.get('content'),
                    "context": hit.entity.get('context')
                },
                "similarity": hit.score
            } for hit in results[0]]
        finally:
            # Always release collection
            self.collection.release()

    def load_existing_json(self, json_data: List[Dict[str, Any]], batch_size: int = 1000):
        """Load existing JSON data with contexts into Milvus."""
        print("Loading existing JSON data into Milvus...")
        entities = []

        for group in tqdm(json_data, desc="Processing groups"):
            for chunk in group['chunks']:
                if 'context' not in chunk:
                    print(f"Skipping chunk {chunk['id']} in group {group['id']} - no context found")
                    continue

                try:
                    combined_text = f"{chunk['text']}\n\n{chunk['context']}"
                    embedding_result = self.voyage_client.embed([combined_text], model="voyage-2")

                    entities.append({
                        "id": chunk['id'],
                        "vector": embedding_result.embeddings[0],
                        "group_id": group['id'],
                        "chunk_id": chunk['id'],
                        "content": chunk['text'],
                        "context": chunk['context']
                    })

                    if len(entities) >= batch_size:
                        self._insert_batch(entities)
                        entities = []

                except Exception as e:
                    print(f"Error processing chunk {chunk['id']} in group {group['id']}: {e}")
                    continue

        if entities:
            self._insert_batch(entities)

        print("Successfully loaded chunks into Milvus")


def process_documents(base_dir: str, db: ContextualVectorDB, parallel_threads: int = 4):
    """Process all documents in the base directory."""
    processed_dir = os.path.join(base_dir, "processed")

    for hash_dir in os.listdir(processed_dir):
        json_path = os.path.join(processed_dir, hash_dir, "grouped_pages.json")
        if not os.path.exists(json_path):
            continue

        print(f"Processing {hash_dir}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        if any('context' in chunk
               for group in json_data
               for chunk in group['chunks']):
            print(f"Skipping {hash_dir} - context already exists")
            continue

        print(f"Processing {hash_dir}...")
        db.load_data(json_data, json_path, parallel_threads)


if __name__ == "__main__":
    # Initialize Milvus database
    db = ContextualVectorDB("pdf_embeddings")

    # Process JSONs from hash directories
    base_dir = "../DOCS"
    processed_dir = os.path.join(base_dir, "processed")

    for hash_dir in os.listdir(processed_dir):
        json_path = os.path.join(processed_dir, hash_dir, "grouped_pages.json")
        if os.path.exists(json_path):
            print(f"Processing {hash_dir}...")
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

                # Check if context already exists in any chunk
                context_exists = any(
                    'context' in chunk
                    for group in json_data
                    for chunk in group['chunks']
                )
                if context_exists:
                    print(f"Skipping {hash_dir} - context already exists")
                    continue

                print(f"Processing {hash_dir}...")
                db.load_data(json_data, json_path, parallel_threads=4)

    # Example search
    results = db.search("shitij", k=5)
    for result in results:
        print(f"Similarity: {result['similarity']:.3f}")
        print(f"Group ID: {result['metadata']['group_id']}")
        print(f"Chunk ID: {result['metadata']['chunk_id']}")
        print(f"Content: {result['metadata']['original_content'][:200]}...")
        print(f"Context: {result['metadata']['context']}\n")
