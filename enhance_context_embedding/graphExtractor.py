import json
import os
from typing import List

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
        "description": (
            "Extracts distinct entities using comprehensive guidelines. "
            "Extract each entity with properties: unique id, canonical name, primary category, "
            "optional subtypes, attributes (aliases, temporal context, domain context, hierarchical level), "
            "and evidence (text snippets, positions, confidence)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity_id": {"type": "string"},
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                            "subtypes": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "attributes": {
                                "type": "object",
                                "properties": {
                                    "aliases": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "temporal_context": {"type": "string"},
                                    "domain_context": {"type": "string"},
                                    "hierarchical_level": {"type": "string"}
                                }
                            },
                            "evidence": {
                                "type": "object",
                                "properties": {
                                    "text_snippets": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "positions": {
                                        "type": "array",
                                        "items": {
                                            "type": "array",
                                            "items": {"type": "number"}
                                        }
                                    },
                                    "confidence": {"type": "number"}
                                }
                            }
                        },
                        "required": ["entity_id", "name", "type", "evidence"]
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
        "description": (
            "Extracts relationships using comprehensive guidelines. "
            "For each relation, include a unique relation id, source entity, target entity, primary relationship type, "
            "optional subtypes, properties (direction, temporal context, strength, certainty), evidence (text snippets, "
            "positions, inference chain), metadata (domain context, extraction method, confidence)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "relations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "relation_id": {"type": "string"},
                            "source_entity": {"type": "string"},
                            "target_entity": {"type": "string"},
                            "type": {"type": "string"},
                            "subtypes": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "properties": {
                                "type": "object",
                                "properties": {
                                    "direction": {"type": "string"},
                                    "temporal_context": {
                                        "type": "object",
                                        "properties": {
                                            "start": {"type": "string"},
                                            "end": {"type": "string"},
                                            "duration": {"type": "string"}
                                        }
                                    },
                                    "strength": {"type": "number"},
                                    "certainty": {"type": "number"}
                                }
                            },
                            "evidence": {
                                "type": "object",
                                "properties": {
                                    "text_snippets": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "positions": {
                                        "type": "array",
                                        "items": {
                                            "type": "array",
                                            "items": {"type": "number"}
                                        }
                                    },
                                    "inference_chain": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                }
                            },
                            "metadata": {
                                "type": "object",
                                "properties": {
                                    "domain_context": {"type": "string"},
                                    "extraction_method": {"type": "string"},
                                    "confidence": {"type": "number"}
                                }
                            }
                        },
                        "required": ["relation_id", "source_entity", "target_entity", "type", "evidence", "metadata"]
                    }
                }
            },
            "required": ["relations"]
        }
    }
]


# ---------------------------------------------------------------------------
# Extraction Functions Using Tool Technique with Generalized Prompts
# ---------------------------------------------------------------------------
def extract_entities_with_tool(doc_content: str) -> list:
    """
    Uses the extract_entities tool with a generalized prompt to extract distinct entities.
    Returns a list of entity names (or a simplified version if you wish to ignore additional details).
    """

    prompts = {
        "cache-prompt": f"""
        Text:
            <document>
            {doc_content}
            </document>
        """,
        "prompt": f"""
                As an entity extraction specialist, analyze the provided text and identify all meaningful entities.
                
                Primary Task:
                Extract distinct entities that are significant within the context, considering:
                - Complete and precise entity mentions
                - Both explicit and implicit entities
                - Canonical forms and aliases
                - Hierarchical relationships and domain-specific significance
                
                Expected Output Format:
                {{
                    "entities": [
                        {{
                            "entity_id": "unique_identifier",
                            "name": "canonical_name",
                            "type": "primary_category",
                            "subtypes": ["more_specific_categories"],
                            "attributes": {{
                                "aliases": ["alternative_names"],
                                "temporal_context": "relevant_time_period",
                                "domain_context": "specific_field",
                                "hierarchical_level": "position_in_hierarchy"
                            }},
                            "evidence": {{
                                "text_snippets": ["supporting_text"],
                                "positions": [[start, end]],
                                "confidence": 0.0
                            }}
                        }}
                    ]
                }}
                
                """
    }
    try:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            tools=tools_for_entities,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompts["cache-prompt"], "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": prompts["prompt"]}
                ]
            }],
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )
        json_entities = None
        for content in response.content:
            if content.type == "tool_use" and content.name == "extract_entities":
                json_entities = content.input  # Expected to be a dict with key "entities"
                break
        # For simplicity, if you only need the canonical names, you can extract them:
        if json_entities and "entities" in json_entities:
            return [item["name"] for item in json_entities["entities"]]
        else:
            print("No valid entities extracted.")
            return []
    except Exception as e:
        print("Error during tool-based entity extraction:", e)
        return []


def extract_relations_with_tool(chunk_text: str, doc_content: str, entities: List[str]) -> list:
    """
    Uses the extract_relations tool with a generalized prompt to extract relationships.
    Returns a list of (source_entity, target_entity, relation) tuples.
    """
    prompts = {
        "cache-prompt": f"""
        Text:
            <document>
            {doc_content}
            </document>
        """,
        "prompt": f"""
        As a relation extraction specialist, analyze the provided text and identify meaningful relationships between entities.
        
        Primary Task:
        Extract relationships between entities that represent significant connections, considering:
        - Both explicit and implicit relationships
        - Directional nature and strength
        - Temporal aspects and conditional relationships
        
        Expected Output Format:
        {{
            "relations": [
                {{
                    "relation_id": "unique_identifier",
                    "source_entity": "entity_id_1",
                    "target_entity": "entity_id_2",
                    "type": "primary_relationship_type",
                    "subtypes": ["more_specific_types"],
                    "properties": {{
                        "direction": "source_to_target",
                        "temporal_context": {{"start": "start_time", "end": "end_time", "duration": "time_period"}},
                        "strength": 0.0,
                        "certainty": 0.0
                    }},
                    "evidence": {{
                        "text_snippets": ["supporting_text"],
                        "positions": [[start, end]],
                        "inference_chain": ["reasoning_steps"]
                    }},
                    "metadata": {{
                        "domain_context": "specific_field",
                        "extraction_method": "explicit",
                        "confidence": 0.0
                    }}
                }}
            ]
        }}
        Entities:
        <entities>{entities}</entities>

        Chunk Text:
        <text>
        {chunk_text}
        </text>        
        """}
    try:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            tools=tools_for_relations,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompts["cache-prompt"], "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": prompts["prompt"]}
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
            # For simplicity, return a tuple (source_entity, target_entity, relation)
            # Here, we assume source_entity is the document (or chunk) and target_entity is the related entity.
            # You may adapt as needed.
            return [(item["source_entity"], item["target_entity"], item["type"]) for item in
                    json_relations["relations"]]
        else:
            print("No valid relations extracted.")
            return []
    except Exception as e:
        print("Error during tool-based relation extraction:", e)
        return []


# ---------------------------------------------------------------------------
# Process Documents to Build Graph JSON Structure
# ---------------------------------------------------------------------------
def process_documents(input_json_path: str, output_json_path: str):
    """
    Reads an input JSON file (an array of document objects) with fields:
      - id: unique document id.
      - text: full document text.
      - chunks: array of chunks, each with:
          - id: chunk id
          - text: chunk text.
    Uses tool-based extraction to extract top-level entities and chunk-level relations.
    Writes the output JSON with a structured graph representation.
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)

    output_structure = {"docs": []}

    for doc in docs:
        doc_out = {}
        doc_out["id"] = doc["id"]
        doc_out["text"] = doc["text"]
        print(f"Extracting entities for document {doc['id']}...")
        doc_entities = extract_entities_with_tool(doc.get("text", ""))
        doc_out["entities"] = doc_entities

        doc_out["chunks"] = []
        if "chunks" in doc:
            for chunk in doc["chunks"]:
                chunk_out = {}
                chunk_out["id"] = chunk["id"]
                chunk_out["text"] = chunk.get("text", "")
                print(f"Extracting relations for document {doc['id']} chunk {chunk['id']}...")
                relations = extract_relations_with_tool(chunk.get("text", ""), doc.get("text", ""),
                                                        entities=doc_entities)
                # Here, we simplify the tuple to {"source": doc_id, "target": entity, "relation": relation}
                # Adjust according to your needs.
                chunk_out["relations"] = [{"source": doc["id"], "target": target, "relation": rel} for _, target, rel in
                                          relations]
                doc_out["chunks"].append(chunk_out)
        output_structure["docs"].append(doc_out)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_structure, f, indent=2)
    print(f"Graph output written to {output_json_path}")


# ---------------------------------------------------------------------------
# Interactive Graph Visualization using Plotly
# ---------------------------------------------------------------------------
def visualize_graph_interactive(json_path: str, output_file="graph_interactive.html"):
    """
    Loads the graph JSON structure (as produced by process_documents) and creates an interactive Plotly graph.
    The graph includes:
      - Document nodes (id = doc id)
      - Chunk nodes (id = "docID-chunkID")
      - Entity nodes (id = entity name)
    Edges:
      - From document to top-level entities (labeled "mentions")
      - From document to chunks (labeled "has_chunk")
      - From chunks to related entities (labeled with the extracted relation)
    The resulting interactive graph is saved as an HTML file.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Build a NetworkX DiGraph first.
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

    # Process documents using tool-based extraction to build the graph JSON.
    process_documents(input_json_path, graph_output_path)

    # Visualize the resulting graph interactively using Plotly.
    G = visualize_graph_interactive(graph_output_path, output_file="graph_interactive.html")

    # Query the graph.
    query_term = input("Enter a node id or label to query the graph: ")
    query_graph(G, query_term)
