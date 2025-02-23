# vector_db.py

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import anthropic
import matplotlib
import voyageai
from pymilvus import Collection, connections, CollectionSchema, FieldSchema, DataType
from pymilvus.orm import utility
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt
from tqdm import tqdm

matplotlib.use("Agg")  # for saving static images without a display


# ------------------------------ VectorDB Class ---------------------------------
class ContextualVectorDB:
    def __init__(self, collection_name: str, voyage_api_key=None, anthropic_api_key=None):
        """Initialize ContextualVectorDB with API clients, Milvus setup, and extraction components."""
        self.voyage_client = voyageai.Client(api_key=voyage_api_key or os.getenv("VOYAGE_API_KEY"))
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.collection_name = collection_name
        self.dimension = 1024
        self.MODEL_NAME = "claude-3-haiku-20240307"
        self._setup_collection()

        self.token_counts = {'input': 0, 'output': 0, 'cache_read': 0, 'cache_creation': 0}
        self.token_lock = threading.Lock()
        self.requests_per_minute = 45
        self.last_request_time = 0
        self.request_lock = threading.Lock()

    def _wait_for_rate_limit(self):
        with self.request_lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            if elapsed < 60 / self.requests_per_minute:
                time.sleep((60 / self.requests_per_minute) - elapsed)
            self.last_request_time = time.time()

    def _setup_collection(self):
        connections.connect(host="localhost", port="19530")
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False, description="primary id"),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension, description="embedding vector"),
            FieldSchema(name="group_id", dtype=DataType.INT64, description="group identifier"),
            FieldSchema(name="chunk_id", dtype=DataType.INT64, description="chunk identifier"),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535, description="chunk content"),
            FieldSchema(name="context", dtype=DataType.VARCHAR, max_length=65535, description="chunk context")
        ]
        schema = CollectionSchema(fields=fields, description="PDF chunks with context", enable_dynamic_field=True)
        self.collection = Collection(name=self.collection_name, schema=schema, using='default')
        index_params = {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 1024}}
        self.collection.create_index(field_name="vector", index_params=index_params)

    # ---------------------- Extraction Tools & Functions -------------------------
    # Tools for entities and relations (using generalized prompts) – we assume these are defined externally.
    # ---------------------------------------------------------------------------
    tools_for_entities = [
        {
            "name": "extract_entities",
            "description": "Extracts distinct entities from the top-level document text.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entity": {
                                    "type": "string",
                                    "description": "The extracted entity name."
                                }
                            },
                            "required": ["entity"]
                        }
                    }
                },
                "required": ["entities"]
            }
        }
    ]

    tools_for_relations = [
        {
            "name": "extract_relations",
            "description": "Extracts relationships from the chunk text.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entity": {
                                    "type": "string",
                                    "description": "The related entity."
                                },
                                "relation": {
                                    "type": "string",
                                    "description": "The type of relationship."
                                }
                            },
                            "required": ["entity", "relation"]
                        }
                    }
                },
                "required": ["relations"]
            }
        }
    ]

    @retry(
        retry=retry_if_exception_type(anthropic.RateLimitError),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3)
    )
    def extract_entities_with_tool(self, doc_content: str) -> list:
        """
        Uses the extract_entities tool to extract distinct entities from the given text.
        Returns a list of entity names.
        """
        prompts = {
            'document': f"<document>\n{doc_content}\n</document>",
            'query': "Extracts entities from document."
        }
        try:
            response = self.anthropic_client.messages.create(
                model=self.MODEL_NAME,
                max_tokens=1024,
                tools=self.tools_for_entities,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompts['query']},
                        {"type": "text", "text": prompts['document']},
                    ]
                }],
            )
            json_entities = None
            for content in response.content:
                if content.type == "tool_use" and content.name == "extract_entities":
                    json_entities = content.input  # Expected to be a dict with key "entities"
                    break

            # If json_entities is a string, parse it into a dict.
            if isinstance(json_entities, str):
                try:
                    json_entities = json.loads(json_entities)
                except Exception as parse_error:
                    print("Failed to parse json_entities:", parse_error)
                    return []

            if json_entities and "entities" in json_entities:
                return [item["entity"] for item in json_entities["entities"]]
            else:
                print("No valid entities extracted.")
                return []
        except Exception as e:
            print("Error during tool-based entity extraction:", e)
            return []

    @retry(
        retry=retry_if_exception_type(anthropic.RateLimitError),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3)
    )
    def extract_relations_with_tool(self, chunk_text: str, entities: List[str]) -> list:
        """
        Uses the extract_relations tool to extract relationships from the given text.
        Returns a list of (entity, relation) tuples.
        """
        prompts = {
            'query': f"Extracts relationships from text. take the reference from the entities provided if "
                     f"required.\n<text>\n{chunk_text}\n</text>",
            'entities': f"Extracts entities from text.\n<entities>{entities}\n</entities>",
        }
        try:
            response = self.anthropic_client.messages.create(
                model=self.MODEL_NAME,
                max_tokens=1024,
                tools=self.tools_for_relations,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompts['entities'],
                            "cache_control": {"type": "ephemeral"}
                        },
                        {
                            "type": "text",
                            "text": prompts['query']
                        }
                    ]
                }],
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
            )
            json_relations = None
            for content in response.content:
                if content.type == "tool_use" and content.name == "extract_relations":
                    json_relations = content.input  # Expected to be a dict with key "relations"
                    break
            if json_relations and "relations" in json_relations:
                return [(item["entity"], item["relation"]) for item in json_relations["relations"]]
            else:
                print("No valid relations extracted.")
                return []
        except Exception as e:
            print("Error during tool-based relation extraction:", e)
            return []

    @retry(
        retry=retry_if_exception_type(anthropic.RateLimitError),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3)
    )
    def generate_context(self, doc_text: str, chunk_text: str) -> str:
        """
        Generates context for a chunk using Claude.
        Returns a succinct context string.
        """
        prompts = {
            'document': f"<document>\n{doc_text}\n</document>",
            'chunk': """Here is the chunk we want to situate within the whole document:
    <chunk>
    {chunk_text}
    </chunk>
    Please provide a short, succinct context to situate this chunk within the overall document.
    Answer only with the context and nothing else."""
        }
        try:
            response = self.anthropic_client.messages.create(
                model=self.MODEL_NAME,
                max_tokens=4096,
                temperature=0.0,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompts['document'],
                            "cache_control": {"type": "ephemeral"}
                        },
                        {
                            "type": "text",
                            "text": prompts['chunk'].format(chunk_text=chunk_text)
                        }
                    ]
                }],
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
            )
            context = response.content[0].text.strip()
            return context
        except Exception as e:
            print("Error during context generation:", e)
            return ""

    # ------------------ Milvus Insertion and Search Methods ---------------------
    def _update_json_file(self, json_path: str, json_lock: threading.Lock,
                          group_id: int, chunk_id: int, context: str):
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

    def _process_chunk(self, group: Dict[str, Any], chunk: Dict[str, Any],
                       json_path: str, json_lock: threading.Lock,
                       doc_entities: List[str]) -> Dict[str, Any]:
        """
        Process a single chunk:
          - Generate context using the document and chunk text.
          - Update the JSON file with the generated context.
          - Combine the chunk text with context to generate an embedding.
          - Extract relations using the generated context, full document text, and pre-extracted document entities.
        Returns a dictionary that merges vector data, original text, generated context, and extracted relations.
        """
        try:
            # Generate context for this chunk.
            context = self.generate_context(group['text'], chunk['text'])
            # Update the JSON file with this generated context.
            self._update_json_file(json_path, json_lock, group['id'], chunk['id'], context)
            # Generate embedding using the combined chunk text and context.
            combined_text = f"{chunk['text']}\n\n{context}"
            embedding_result = self.voyage_client.embed([combined_text], model="voyage-2")
            # Extract relations using the generated context (instead of raw chunk text),
            # along with the document text and pre-extracted document entities.
            relations = self.extract_relations_with_tool(chunk['text'], entities=doc_entities)
            return {
                "id": chunk['id'],
                "vector": embedding_result.embeddings[0],
                "group_id": group['id'],
                "chunk_id": chunk['id'],
                "content": chunk['text'],
                "context": context,
                "relations": [{"source": group['id'], "target": target, "relation": rel}
                              for target, rel in relations]
            }
        except Exception as e:
            print(f"Error processing chunk {chunk['id']} in group {group['id']}: {e}")
            return None

    def load_data(self, json_data: List[Dict[str, Any]], json_path: str, parallel_threads: int = 4) -> Dict[str, Any]:
        """
        Processes the input JSON (an array of document objects) in parallel to generate a combined JSON
        containing document text, document-level entities, each chunk's text, generated context, and extracted relations.
        Also inserts the processed chunks (with embeddings) into Milvus.
        Returns the combined JSON structure.
        """
        total_chunks = sum(len(group['chunks']) for group in json_data)
        json_lock = threading.Lock()
        processed_chunks = []

        combined_json = {"docs": []}

        def process_single_document(doc: dict) -> dict:
            doc_out = {}
            doc_out["id"] = doc["id"]
            doc_out["text"] = doc["text"]
            print(f"Extracting entities for document {doc['id']}...")
            doc_entities = self.extract_entities_with_tool(doc.get("text", ""))
            doc_out["entities"] = doc_entities
            doc_out["chunks"] = []
            if "chunks" in doc:
                with ThreadPoolExecutor(max_workers=parallel_threads) as chunk_executor:
                    future_to_chunk = {
                        chunk_executor.submit(self._process_chunk, doc, chunk, json_path, json_lock,
                                              doc_entities): chunk
                        for chunk in doc["chunks"]
                    }
                    for future in as_completed(future_to_chunk):
                        chunk_out = future.result()
                        doc_out["chunks"].append(chunk_out)
            return doc_out

        with ThreadPoolExecutor(max_workers=parallel_threads) as doc_executor:
            futures = [doc_executor.submit(process_single_document, doc) for doc in json_data]
            for future in as_completed(futures):
                combined_json["docs"].append(future.result())

        # Optionally, insert embeddings into Milvus
        for group in combined_json["docs"]:
            for chunk in group["chunks"]:
                if chunk:
                    processed_chunks.append(chunk)
                    if len(processed_chunks) >= 1000:
                        self._insert_batch(processed_chunks)
                        processed_chunks = []
        if processed_chunks:
            self._insert_batch(processed_chunks)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(combined_json, f, indent=2)
        print(f"Combined graph & context output written to {json_path}")
        return combined_json

    def _insert_batch(self, entities: List[Dict[str, Any]]):
        try:
            batch = []
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
            self.collection.load()
            try:
                self.collection.insert(batch)
                print(f"Inserted {len(entities)} entities")
            finally:
                self.collection.release()
        except Exception as e:
            print(f"Error in batch insert: {e}")
            print(f"First entity structure: {entities[0]}")
            raise

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        query_vector = self.voyage_client.embed([query], model="voyage-2").embeddings[0]
        self.collection.load()
        try:
            results = self.collection.search(
                data=[query_vector],
                anns_field="vector",
                param={"metric_type": "IP", "params": {"nprobe": 10}},
                limit=k,
                output_fields=["group_id", "chunk_id", "content", "context"]
            )
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
            self.collection.release()

    def load_existing_json(self, json_data: List[Dict[str, Any]], batch_size: int = 1000):
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


if __name__ == "__main__":
    db = ContextualVectorDB("pdf_embeddings")
    base_dir = "../DOCS"
    processed_dir = os.path.join(base_dir, "processed")
    for hash_dir in os.listdir(processed_dir):
        json_path = os.path.join(processed_dir, hash_dir, "grouped_pages.json")
        if os.path.exists(json_path):
            print(f"Processing {hash_dir}...")
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                context_exists = any('context' in chunk for group in json_data for chunk in group['chunks'])
                if context_exists:
                    print(f"Skipping {hash_dir} - context already exists")
                    continue
                print(f"Processing {hash_dir}...")
                db.load_data(json_data, json_path, parallel_threads=4)

    results = db.search("shitij", k=5)
    for result in results:
        print(f"Similarity: {result['similarity']:.3f}")
        print(f"Group ID: {result['metadata']['group_id']}")
        print(f"Chunk ID: {result['metadata']['chunk_id']}")
        print(f"Content: {result['metadata']['original_content'][:200]}...")
        print(f"Context: {result['metadata']['context']}\n")
