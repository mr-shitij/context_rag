# reranker.py

import os
import json
from typing import List, Dict, Any
import anthropic

MODEL_NAME = "claude-3-haiku-20240307"

# Define the tool for re-ranking that returns structured output.
tools_for_rerank = [
    {
        "name": "rerank_candidates",
        "description": (
            "Re-ranks candidate chunks based on the provided query. "
            "Input is a prompt that lists each candidate with its snippet and score. "
            "Return a structured JSON object with a key 'ranking' whose value is a comma-separated "
            "list of candidate numbers (1-indexed) in descending order of relevance. "
            "For example: {\"ranking\": \"2,5,1,3,4\"}"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ranking": {
                    "type": "string",
                    "description": "Comma-separated candidate indices in order (1-indexed)"
                }
            },
            "required": ["ranking"]
        }
    }
]


class ReRanker:
    def __init__(self, anthropic_api_key: str = None, model: str = MODEL_NAME):
        self.model = model
        self.client = anthropic.Anthropic(api_key=anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"))

    def _build_rerank_prompt(self, query: str, candidates: List[Dict[str, Any]], source: str) -> str:
        prompt = f"Query: {query}\n\n"
        prompt += f"Below are candidate chunks from the {source} search:\n"
        for i, cand in enumerate(candidates):
            meta = cand.get("metadata", {})
            snippet = meta.get("context") or meta.get("original_content", "")
            score = cand.get("similarity", 0) if source == "vector" else cand.get("graph_score", 0)
            snippet = snippet.strip().replace("\n", " ")[:200]
            prompt += f"{i + 1}. Score: {score:.3f} - {snippet}\n"
        prompt += (
            "\nUsing the rerank_candidates tool, re-rank these candidates from most relevant to least relevant. "
            "Return your answer as a JSON object with a key 'ranking' whose value is a comma-separated list of candidate "
            "numbers (1-indexed). For example: {\"ranking\": \"2,5,1,3,4\"}"
        )
        return prompt

    def _rerank_with_tool(self, query: str, candidates: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
        prompt = self._build_rerank_prompt(query, candidates, source)
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                tools=tools_for_rerank,
                messages=[{"role": "user", "content": prompt}],
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
            )
            output_text = response.content[0].text.strip()
            print(f"LLM re-ranker ({source}) raw output:", output_text)
        except Exception as e:
            print("Error during LLM re-ranking:", e)
            return candidates

        try:
            output_json = json.loads(output_text)
            ranking_str = output_json.get("ranking", "")
            indices = [int(i.strip()) for i in ranking_str.split(",") if i.strip().isdigit()]
        except Exception as parse_err:
            print("Error parsing re-ranker output:", parse_err)
            indices = list(range(1, len(candidates) + 1))

        ranked_candidates = [candidates[idx - 1] for idx in indices if 1 <= idx <= len(candidates)]
        return ranked_candidates

    def rank_vector_candidates(self, query: str, vector_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ranked = self._rerank_with_tool(query, vector_candidates, source="vector")
        return ranked[:5]

    def rank_graph_candidates(self, query: str, graph_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ranked = self._rerank_with_tool(query, graph_candidates, source="graph")
        return ranked[:5]

    def merge_and_rerank(self,
                         query: str,
                         vector_candidates: List[Dict[str, Any]],
                         graph_candidates: List[Dict[str, Any]],
                         weights: Dict[str, float] = {"vector": 0.6, "graph": 0.4}) -> List[Dict[str, Any]]:
        """
        Re-ranks vector and graph candidates separately using the LLM and then merges the two sets.
        Each candidate is assumed to have:
            - "similarity" score for vector candidates,
            - "graph_score" for graph candidates (or defaults to 0).
        The final score is computed as:
            final_score = weights["vector"] * similarity + weights["graph"] * graph_score.
        Returns the top 10 merged candidates.
        """
        top_vector = self.rank_vector_candidates(query, vector_candidates)
        top_graph = self.rank_graph_candidates(query, graph_candidates)
        merged = top_vector + top_graph

        for cand in merged:
            sim = cand.get("similarity", 0)
            gscore = cand.get("graph_score", 0)
            cand["final_score"] = weights["vector"] * sim + weights["graph"] * gscore

        merged_sorted = sorted(merged, key=lambda x: x["final_score"], reverse=True)
        return merged_sorted[:10]
