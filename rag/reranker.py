import os
import json
from typing import List, Dict, Any

import anthropic

MODEL_NAME = "claude-3-haiku-20240307"


class ReRanker:
    def __init__(self, anthropic_api_key: str = None, model: str = MODEL_NAME):
        """
        Initialize the LLM-based re-ranker using Anthropic's API.
        """
        self.model = model
        self.client = anthropic.Anthropic(api_key=anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"))

    def _build_rerank_prompt(self, query: str, candidates: list) -> str:
        """
        Flattens the candidate structure and builds a re-ranking prompt.

        For each candidate, it extracts:
          - The candidate ID in the format "docID-chunkID"
          - The similarity score (from the "similarity" field)
          - A snippet from either "context" or "original_content" in the candidate metadata.

        The prompt instructs the LLM to return a JSON object with a key "ranking" whose value is a comma-separated list of
        candidate IDs (docID-chunkID) ordered from most relevant to least.
        """
        prompt = f"Query: {query}\n\n"
        prompt += "Below are candidate chunks retrieved from the vector-graph search. Each candidate is identified by a candidate ID in the format 'docID-chunkID'.\n\n"
        for i, cand in enumerate(candidates):
            metadata = cand.get("metadata", {})
            group_id = str(metadata.get("group_id", ""))
            chunk_id = str(metadata.get("chunk_id", ""))
            candidate_id = f"{group_id}-{chunk_id}"
            # Use context as the snippet if available; fallback to original_content.
            snippet = (metadata.get("context") or metadata.get("original_content") or "").replace("\n", " ")[
                      :200].strip()
            prompt += f"{i + 1}. ID: {candidate_id} - Snippet: {snippet}\n"
        prompt += (
            "\nRe-rank these candidates from most relevant to least relevant. "
            "Return your answer as a JSON object with a key 'ranking' whose value is a comma-separated list of candidate IDs (docID-chunkID). "
            "For example: {\"ranking\": \"1-2,1-3,2-1,...\"}. only give the ranking JSON nothing else."
        )
        return prompt

    def rerank_candidates(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Uses the LLM to re-rank a flattened candidate list.

        Expects the LLM to output a JSON string like:
          {"ranking": "1-2,2-1,1-3,..."}
        where each candidate ID is in the format "docID-chunkID".

        Returns the candidate list re-ordered according to the LLM output.
        """
        prompt = self._build_rerank_prompt(query, candidates)
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            output_text = response.content[0].text.strip()
            print("LLM re-ranker raw output:", output_text)
        except Exception as e:
            print("Error during LLM re-ranking:", e)
            return candidates

        try:
            # Parse the LLM response. Expected output: {"ranking": "doc1-chunk1,doc2-chunk3,..."}
            output_json = json.loads(output_text)
            ranking_str = output_json.get("ranking", "")
            candidate_ids = [cid.strip() for cid in ranking_str.split(",") if cid.strip()]
        except Exception as parse_err:
            print("Error parsing re-ranker output:", parse_err)
            candidate_ids = []

        # Build mapping from candidate ID to candidate.
        candidate_mapping = {}
        for cand in candidates:
            metadata = cand.get("metadata", {})
            group_id = str(metadata.get("group_id", ""))
            chunk_id = str(metadata.get("chunk_id", ""))
            cand_id = f"{group_id}-{chunk_id}"
            candidate_mapping[cand_id] = cand

        ranked_candidates = []
        for cid in candidate_ids:
            if cid in candidate_mapping:
                ranked_candidates.append(candidate_mapping[cid])
        # Append any candidates not mentioned in the LLM output, preserving their original order.
        for cand in candidates:
            metadata = cand.get("metadata", {})
            cand_id = f"{str(metadata.get('group_id', ''))}-{str(metadata.get('chunk_id', ''))}"
            if cand_id not in candidate_ids:
                ranked_candidates.append(cand)
        return ranked_candidates
