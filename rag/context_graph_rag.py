import json
import os
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import matplotlib
import networkx as nx
import plotly.graph_objects as go

matplotlib.use("Agg")  # for saving static images without a display

# Initialize Anthropic client with your API key.
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY",
                                               "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"))
MODEL_NAME = "claude-3-haiku-20240307"

# ---------------------------------------------------------------------------
# Define Tools for Extraction with Generalized Prompts
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


# ---------------------------------------------------------------------------
# Extraction Functions Using Tool Technique (with prompt caching enabled)
# ---------------------------------------------------------------------------
def extract_entities_with_tool(doc_content: str) -> list:
    """
    Uses the extract_entities tool to extract distinct entities from the given text.
    Returns a list of entity names.
    """
    prompts = {
        'document': f"<document>\n{doc_content}\n</document>",
        'query': "Extracts entities from document."
    }
    try:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            tools=tools_for_entities,
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


def extract_relations_with_tool(chunk_text: str, entities: List[str]) -> list:
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
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            tools=tools_for_relations,
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


# ---------------------------------------------------------------------------
# Context Generation Function
# ---------------------------------------------------------------------------
def generate_context(doc_text: str, chunk_text: str) -> str:
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
        response = client.messages.create(
            model=MODEL_NAME,
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


def process_documents(input_json_path: str, output_json_path: str, parallel_threads: int = 4):
    """
    Reads an input JSON file (an array of document objects) with fields:
      - id: unique document id.
      - text: full document text.
      - chunks: array of chunks, each with:
          - id: chunk id
          - text: chunk text.
    Uses tool-based extraction to extract top-level entities and chunk-level relations.
    Writes the output JSON with a structured graph representation incrementally
    (checkpointing after processing each document) so that in case of failure the process can resume.
    """
    # Load input documents
    with open(input_json_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)

    # Try to load existing checkpoint, if it exists.
    if os.path.exists(output_json_path):
        with open(output_json_path, 'r', encoding='utf-8') as f:
            output_structure = json.load(f)
        processed_ids = {doc["id"] for doc in output_structure.get("docs", [])}
        print(f"Resuming from checkpoint. Already processed document ids: {processed_ids}")
    else:
        output_structure = {"docs": []}
        processed_ids = set()

    # Define functions for processing a single document and a single chunk (same as before)
    def process_single_document(doc: dict) -> dict:
        # If already processed, simply return None to skip.
        if doc["id"] in processed_ids:
            return None
        doc_out = {}
        doc_out["id"] = doc["id"]
        doc_out["text"] = doc["text"]
        print(f"Extracting entities for document {doc['id']}...")
        doc_entities = extract_entities_with_tool(doc.get("text", ""))
        doc_out["entities"] = doc_entities

        doc_out["chunks"] = []
        if "chunks" in doc:
            with ThreadPoolExecutor(max_workers=parallel_threads) as chunk_executor:
                future_to_chunk = {
                    chunk_executor.submit(process_single_chunk, doc, chunk, doc_entities): chunk for chunk in
                    doc["chunks"]
                }
                for future in as_completed(future_to_chunk):
                    chunk_out = future.result()
                    doc_out["chunks"].append(chunk_out)
        return doc_out

    def process_single_chunk(doc: dict, chunk: dict, doc_entities: List[str]) -> dict:
        chunk_out = {}
        chunk_out["id"] = chunk["id"]
        chunk_out["text"] = chunk.get("text", "")
        print(f"Generating context for document {doc['id']} chunk {chunk['id']}...")
        generated_context = generate_context(doc.get("text", ""), chunk.get("text", ""))
        chunk_out["context"] = generated_context
        print(f"Extracting relations for document {doc['id']} chunk {chunk['id']}...")
        relations = extract_relations_with_tool(generated_context, entities=doc_entities)
        # Since our extract_relations_with_tool returns tuples (target, relation), adjust accordingly:
        chunk_out["relations"] = [{"source": doc["id"], "target": target, "relation": rel} for target, rel in relations]
        return chunk_out

    # Process documents sequentially (or in parallel, but then careful checkpointing is needed).
    for doc in docs:
        if doc["id"] in processed_ids:
            continue
        processed_doc = process_single_document(doc)
        if processed_doc is not None:
            output_structure["docs"].append(processed_doc)
            processed_ids.add(doc["id"])
            # Write checkpoint after each document is processed.
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(output_structure, f, indent=2)
            print(f"Checkpoint updated: processed document {doc['id']}")

    print(f"Graph output written to {output_json_path}")


# ---------------------------------------------------------------------------
# Interactive Graph Visualization using Plotly
# ---------------------------------------------------------------------------
def visualize_graph_interactive(json_path: str, output_file="graph_interactive.html"):
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
        G.add_node(doc_id, label=f"Doc {doc_id}", type="doc", text=doc["text"][:100])
        for entity in doc.get("entities", []):
            if not G.has_node(entity):
                G.add_node(entity, label=entity, type="entity")
            G.add_edge(doc_id, entity, relation="mentions")
        for chunk in doc.get("chunks", []):
            chunk_node = f"{doc_id}-{chunk['id']}"
            G.add_node(chunk_node, label=f"Chunk {chunk['id']}", type="chunk", text=chunk["text"][:100])
            G.add_edge(doc_id, chunk_node, relation="has_chunk")
            for rel in chunk.get("relations", []):
                target = rel["target"]
                if not G.has_node(target):
                    G.add_node(target, label=target, type="entity")
                G.add_edge(chunk_node, target, relation=rel["relation"])

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


# ---------------------------------------------------------------------------
# Graph Query Function
# ---------------------------------------------------------------------------
def query_graph(G: nx.DiGraph, query: str):
    """
    Queries the graph for a given node (by id or a substring in its label)
    and prints its outgoing and incoming neighbors along with relation labels.
    """
    found_node = None
    if query in G.nodes:
        found_node = query
    else:
        for node, attr in G.nodes(data=True):
            if "label" in attr and query.lower() in attr["label"].lower():
                found_node = node
                break

    if found_node is None:
        print(f"No node found for query: {query}")
        return

    print(f"Query result for node '{found_node}':")
    print("Outgoing neighbors:")
    for neighbor in G.successors(found_node):
        relation = G.edges[found_node, neighbor].get("relation", "")
        print(f"  {found_node} --({relation})--> {neighbor}")
    print("Incoming neighbors:")
    for neighbor in G.predecessors(found_node):
        relation = G.edges[neighbor, found_node].get("relation", "")
        print(f"  {neighbor} --({relation})--> {found_node}")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # File paths: adjust as needed.
    input_json_path = "../DOCS/processed/8d666fe5820af800c8778b001c37c7169b5edb617f42158ca8dcad28fc8d59aa/grouped_pages.json"
    graph_output_path = "graph_output.json"  # Output JSON file for the structured graph.

    # Process documents using tool-based extraction to build the combined graph & context JSON.
    # process_documents(input_json_path, graph_output_path)

    # Visualize the resulting graph interactively using Plotly.
    G = visualize_graph_interactive(graph_output_path, output_file="graph_interactive.html")

    # Query the graph.
    query_term = input("Enter a node id or label to query the graph: ")
    query_graph(G, query_term)
