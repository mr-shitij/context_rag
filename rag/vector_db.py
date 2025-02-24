# vector_db.py

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
import json
import networkx as nx
import plotly.graph_objects as go
from neo4j import GraphDatabase

matplotlib.use("Agg")  # for saving static images without a display


def visualize_graph_interactive(json_path: str, output_file="graph_interactive.html") -> nx.DiGraph:
    """
    Loads the combined JSON structure (produced by process_documents) and creates an interactive Plotly graph.
    The graph includes:
      - Document nodes (id = doc id)
      - Chunk nodes (id = "docID-chunkID") with context stored in metadata
      - Entity nodes (id = entity name)
    Edges:
      - From document to top-level entities (labeled "mentions")
      - From document to chunks (labeled "has_chunk")
      - From chunks to related entities (labeled with the extracted relation)
    The interactive graph is saved as an HTML file.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    G = nx.DiGraph()

    for doc in data.get("docs", []):
        doc_id = doc["id"]
        G.add_node(doc_id, label=f"Doc {doc_id}", type="doc", text=doc.get("text", "")[:100])
        for entity in doc.get("entities", []):
            if not G.has_node(entity):
                G.add_node(entity, label=entity, type="entity")
            G.add_edge(doc_id, entity, relation="mentions")
        for chunk in doc.get("chunks", []):
            # Use .get("text", "") to handle missing 'text' keys.
            chunk_text = chunk.get("content", "")
            chunk_node = f"{doc_id}-{chunk.get('id')}"
            G.add_node(chunk_node, label=f"Chunk {chunk.get('id')}", type="chunk", text=chunk_text[:100])
            G.add_edge(doc_id, chunk_node, relation="has_chunk")
            for rel in chunk.get("relations", []):
                target = rel.get("target")
                if target and not G.has_node(target):
                    G.add_node(target, label=target, type="entity")
                if target:
                    G.add_edge(chunk_node, target, relation=rel.get("relation", ""))

    pos = nx.spring_layout(G, k=0.5, iterations=50)
    edge_x = []
    edge_y = []
    for u, v, data_edge in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    node_x = []
    node_y = []
    node_labels = nx.get_node_attributes(G, 'label')
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[node_labels.get(n, n) for n in G.nodes()],
        textposition="bottom center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[len(list(G.adj[n])) for n in G.nodes()],
            size=20,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left'
            ),
            line_width=2
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Interactive Graph Visualization',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper"
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    fig.write_html(output_file)
    print(f"Interactive graph saved as {output_file}")
    return G


# ------------------------------ VectorDB Class ---------------------------------
class ContextualVectorDB:
    def __init__(self, collection_name: str, voyage_api_key=None, anthropic_api_key=None, neo4j_uri=None,
                 neo4j_user=None, neo4j_password=None):
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

        self.neo4j_uri = voyageai.Client(api_key=neo4j_uri or os.getenv("NEO4J_URI"))
        self.neo4j_user = anthropic.Anthropic(api_key=neo4j_user or os.getenv("NEO4J_USERNAME"))
        self.neo4j_password = anthropic.Anthropic(api_key=neo4j_password or os.getenv("NEO4J_PASSWORD"))
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

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
                        {"type": "text", "text": prompts['document'], "cache_control": {"type": "ephemeral"}},
                        {"type": "text", "text": prompts['query']},
                    ]
                }],
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
            )
            json_entities = None
            for content in response.content:
                if content.type == "tool_use" and content.name == "extract_entities":
                    json_entities = content.input  # This might be a string.
                    break

            if isinstance(json_entities, str):
                try:
                    json_entities = json.loads(json_entities)
                except Exception as parse_error:
                    print("Error parsing json_entities:", parse_error)
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
            context = self.generate_context(group['text'], chunk['content'])
            # Update the JSON file with this generated context.
            self._update_json_file(json_path, json_lock, group['id'], chunk['id'], context)
            # Generate embedding using the combined chunk text and context.
            combined_text = f"{chunk['content']}\n\n{context}"
            embedding_result = self.voyage_client.embed([combined_text], model="voyage-2")
            # Extract relations using the generated context (instead of raw chunk text),
            # along with the document text and pre-extracted document entities.
            relations = self.extract_relations_with_tool(chunk['content'], entities=doc_entities)
            return {
                "id": chunk['id'],
                "vector": embedding_result.embeddings[0],
                "group_id": group['id'],
                "chunk_id": chunk['id'],
                "content": chunk['content'],
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

    def search_neo4j(self, search_string: str):
        """
        Queries the Neo4j graph database for nodes whose 'label' or 'context' contains the search string (case-insensitive),
        calculates a simple matching score, and prints the node along with its outgoing and incoming relationships.

        The matching score is computed as:
          score = (1 if label contains search_string else 0) + (1 if context contains search_string else 0)

        Parameters:
            search_string (str): The string to search for in node labels or context.
        """
        query = """
        MATCH (n:Node)
        WHERE toLower(n.label) CONTAINS toLower($search)
           OR (n.context IS NOT NULL AND toLower(n.context) CONTAINS toLower($search))
        OPTIONAL MATCH (n)-[r]->(m:Node)
        OPTIONAL MATCH (p:Node)-[r2]->(n)
        WITH n,
             collect(DISTINCT {neighbor: m, relation: r.relation}) AS outgoing,
             collect(DISTINCT {neighbor: p, relation: r2.relation}) AS incoming,
             CASE WHEN toLower(n.label) CONTAINS toLower($search) THEN 1 ELSE 0 END AS labelMatch,
             CASE WHEN n.context IS NOT NULL AND toLower(n.context) CONTAINS toLower($search) THEN 1 ELSE 0 END AS contextMatch
        RETURN n, outgoing, incoming, (labelMatch + contextMatch) AS score
        """
        with self.neo4j_driver.session() as session:
            results = session.run(query, search=search_string)
            found = False
            for record in results:
                found = True
                node = record["n"]
                outgoing = record["outgoing"]
                incoming = record["incoming"]
                score = record["score"]
                print(f"Node Found: {node['id']} | Label: {node['label']} | Score: {score}")
                print("Outgoing Connections:")
                for out in outgoing:
                    if out["neighbor"]:
                        print(
                            f"  {node['id']} --({out['relation']})--> {out['neighbor']['id']} (Label: {out['neighbor']['label']})")
                print("Incoming Connections:")
                for inc in incoming:
                    if inc["neighbor"]:
                        print(
                            f"  {inc['neighbor']['id']} --({inc['relation']})--> {node['id']} (Label: {inc['neighbor']['label']})")
                print("=" * 50)
            if not found:
                print("No matching nodes found.")

    def load_existing_json(self, json_data: List[Dict[str, Any]], batch_size: int = 1000):
        print("Loading existing JSON data into Milvus...")
        entities = []
        for group in tqdm(json_data, desc="Processing groups"):
            for chunk in group['chunks']:
                if 'context' not in chunk:
                    print(f"Skipping chunk {chunk['id']} in group {group['id']} - no context found")
                    continue
                try:
                    combined_text = f"{chunk['content']}\n\n{chunk['context']}"
                    embedding_result = self.voyage_client.embed([combined_text], model="voyage-2")
                    entities.append({
                        "id": chunk['id'],
                        "vector": embedding_result.embeddings[0],
                        "group_id": group['id'],
                        "chunk_id": chunk['id'],
                        "content": chunk['content'],
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

    def store_graph_in_neo4j(self, json_path: str):
        """
        Loads the combined JSON structure and creates a NetworkX graph with detailed metadata.
        Then stores the graph into a Neo4j database.

        Expected JSON structure:
        {
          "docs": [
            {
              "id": "document_id",
              "text": "Full document text...",
              "entities": [ "Entity1", "Entity2", ... ],
              "chunks": [
                {
                  "id": "chunk_id",
                  "text": "Chunk text...",
                  "context": "Generated context...",
                  "relations": [
                      { "source": "document_id", "target": "Entity1", "relation": "mentions" },
                      ...
                  ]
                },
                ...
              ]
            },
            ...
          ]
        }

        Each node is created/merged with properties:
          - id: the unique identifier (for docs, chunks, or entities)
          - label: a display label
          - type: node type ("doc", "chunk", "entity")
          - text: a snippet or the full text (as needed)
          - additional metadata (e.g. full_text for documents, context for chunks, etc.)

        Relationships are created with a "relation" property.
        """
        # Load JSON data.
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Build the NetworkX graph with proper metadata.
        G = nx.DiGraph()
        for doc in data.get("docs", []):
            doc_id = doc["id"]
            # Create document node with additional metadata.
            G.add_node(doc_id,
                       label=f"Doc {doc_id}",
                       type="doc",
                       text=doc["text"],
                       entities=doc.get("entities", []))
            # Connect document to its entities.
            for entity in doc.get("entities", []):
                if not G.has_node(entity):
                    G.add_node(entity, label=entity, type="entity")
                G.add_edge(doc_id, entity, relation="mentions")
            # Process chunks.
            for chunk in doc.get("chunks", []):
                chunk_id = chunk.get("id")
                chunk_text = chunk.get("content", "")
                chunk_context = chunk.get("context", "")
                chunk_node = f"{doc_id}-{chunk_id}"
                # Create chunk node with context metadata.
                G.add_node(chunk_node,
                           label=f"Chunk {chunk_id}",
                           type="chunk",
                           full_content=chunk_text,
                           full_context=chunk_context)
                G.add_edge(doc_id, chunk_node, relation="has_chunk")
                for rel in chunk.get("relations", []):
                    target = rel.get("target")
                    relation_label = rel.get("relation", "")
                    if target:
                        if not G.has_node(target):
                            G.add_node(target, label=target, type="entity")
                        G.add_edge(chunk_node, target, relation=relation_label)

        # Now, store the graph into Neo4j.
        with self.neo4j_driver.session() as session:
            # Create or merge nodes.
            for node, props in G.nodes(data=True):
                # We'll store all properties as node attributes.
                session.run(
                    """
                    MERGE (n:Node {id: $id})
                    SET n += $props
                    """,
                    id=node,
                    props=props
                )
            # Create or merge relationships.
            for u, v, props in G.edges(data=True):
                session.run(
                    """
                    MATCH (a:Node {id: $u}), (b:Node {id: $v})
                    MERGE (a)-[r:RELATION {relation: $relation}]->(b)
                    """,
                    u=u,
                    v=v,
                    relation=props.get("relation", "")
                )
        self.neo4j_driver.close()
        print("Graph successfully stored in Neo4j.")
        return G


if __name__ == "__main__":
    db = ContextualVectorDB(
        collection_name="pdf_embeddings",
        voyage_api_key="pa-QhwbHHG0NSWxFv1uw-0KReqcnG8_kjCT8K1OOj3sKf8",
        anthropic_api_key="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",
        neo4j_uri="neo4j+s://7a45783a.databases.neo4j.io",
        neo4j_user="neo4j",
        neo4j_password="UGoQsOUsYYLgdW1VoyTi5QuZEtfU45YIxhFXuNT0gP0"
    )
    # base_dir = "../DOCS"
    # processed_dir = os.path.join(base_dir, "processed")
    # for hash_dir in os.listdir(processed_dir):
    #     json_path = os.path.join(processed_dir, hash_dir, "grouped_pages.json")
    #     if os.path.exists(json_path):
    #         print(f"Processing {hash_dir}...")
    #         with open(json_path, 'r', encoding='utf-8') as f:
    #             json_data = json.load(f)
    #             context_exists = any('context' in chunk for group in json_data for chunk in group['chunks'])
    #             if context_exists:
    #                 print(f"Skipping {hash_dir} - context already exists")
    #                 continue
    #             print(f"Processing {hash_dir}...")
    #             db.load_data(json_data, json_path, parallel_threads=4)
    #
    # results = db.search("Shitij Agrawal", k=5)
    # for result in results:
    #     print(f"Similarity: {result['similarity']:.3f}")
    #     print(f"Group ID: {result['metadata']['group_id']}")
    #     print(f"Chunk ID: {result['metadata']['chunk_id']}")
    #     print(f"Content: {result['metadata']['original_content'][:200]}...")
    #     print(f"Context: {result['metadata']['context']}\n")

    # visualize_graph_interactive("../DOCS/processed/8d666fe5820af800c8778b001c37c7169b5edb617f42158ca8dcad28fc8d59aa/grouped_pages.json")
    # db.store_graph_in_neo4j("../DOCS/processed/8d666fe5820af800c8778b001c37c7169b5edb617f42158ca8dcad28fc8d59aa/grouped_pages.json")
    db.search_neo4j("MCDM")
