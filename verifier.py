"""Verifier and Audit Module for RAG pipeline.

Implements safety layers to check LLM outputs against source evidence
and logs all executions to an audit trail.
"""

import os
import re
import csv
import time
from datetime import datetime

import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None

# -----------------
# Configuration
# -----------------
DATA_DIR = "datasets"
AUDIT_LOG_FILE = "audit_log.csv"
SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"
SEMANTIC_THRESHOLD = 0.5


class NumericVerifier:
    """Verifies that numbers in the generated text exist in the source chunks."""

    def __init__(self, tolerance: float = 0.05):
        """Initialize with a tolerance for numeric matching (e.g., 0.05 for 5%)."""
        self.tolerance = tolerance
        # Regex to match integers and decimals
        self.num_pattern = re.compile(r'\b\d+(?:\.\d+)?\b')

    def extract_numbers(self, text: str) -> list[float]:
        """Extract all valid numbers from a string."""
        matches = self.num_pattern.findall(text)
        nums = []
        for match in matches:
            try:
                nums.append(float(match))
            except ValueError:
                pass
        return nums

    def verify(self, generated_text: str, retrieved_chunks: list[dict]) -> list[float]:
        """Check if numbers in the generated text are supported by the chunks.

        Returns:
            A list of unsupported numbers found in the generated text.
        """
        # Extract numbers from generated text
        gen_nums = self.extract_numbers(generated_text)
        if not gen_nums:
            return []  # No numbers to verify

        # Extract all numbers from retrieved chunks
        chunk_text = " ".join([chunk.get("note_text", "") for chunk in retrieved_chunks])
        source_nums = self.extract_numbers(chunk_text)

        unsupported_nums = []
        for g_num in gen_nums:
            # Check if there is any source number within the tolerance threshold
            is_supported = False
            for s_num in source_nums:
                # Calculate relative difference or absolute if numbers are close to 0
                if s_num == 0:
                    diff = abs(g_num - s_num)
                else:
                    diff = abs(g_num - s_num) / abs(s_num)
                
                if diff <= self.tolerance:
                    is_supported = True
                    break
            
            if not is_supported:
                unsupported_nums.append(g_num)

        return list(set(unsupported_nums))  # Return unique unsupported numbers


class SemanticVerifier:
    """Uses a cross-encoder or embedding model to check claim consistency."""

    def __init__(self, model_name: str = SEMANTIC_MODEL_NAME, threshold: float = SEMANTIC_THRESHOLD):
        """Initialize the semantic embedding model."""
        self.threshold = threshold
        if SentenceTransformer is not None:
            print(f"  Loading Semantic Verifier ({model_name})...")
            self.model = SentenceTransformer(model_name)
        else:
            print("  Warning: sentence-transformers not installed. Semantic check disabled.")
            self.model = None

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences/claims."""
        # Split by periods, exclamation marks, or question marks followed by a space
        sentences = re.split(r'(?<=[.!?]) +', text.replace('\n', ' '))
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def verify(self, generated_text: str, retrieved_chunks: list[dict]) -> list[str]:
        """Check if claims in the generated text are semantically supported.

        Returns:
            A list of unsupported sentences/claims.
        """
        if self.model is None or not retrieved_chunks or not generated_text.strip():
            return []

        claims = self._split_into_sentences(generated_text)
        if not claims:
            return []

        chunk_texts = [chunk.get("note_text", "") for chunk in retrieved_chunks]

        # Encode claims and chunks
        claim_embeddings = self.model.encode(claims, convert_to_tensor=True)
        chunk_embeddings = self.model.encode(chunk_texts, convert_to_tensor=True)

        # Compute cosine similarity between all claims and all chunks
        # Result shape: (len(claims), len(chunks))
        cosine_scores = util.cos_sim(claim_embeddings, chunk_embeddings)

        unsupported_claims = []
        for i, claim in enumerate(claims):
            # Get the max similarity score for this claim across all chunks
            max_score = float(cosine_scores[i].max())
            if max_score < self.threshold:
                unsupported_claims.append(claim)

        return unsupported_claims


def log_provenance(
    query: str,
    mode: str,
    chunks: list[dict],
    generated_text: str,
    unsupported_numbers: list[float],
    unsupported_claims: list[str],
    retrieval_ms: int,
    generation_ms: int,
) -> None:
    """Log the execution details to the audit CSV for traceability."""
    os.makedirs(DATA_DIR, exist_ok=True)
    log_path = os.path.join(DATA_DIR, AUDIT_LOG_FILE)
    
    file_exists = os.path.isfile(log_path)
    
    # Extract just the IDs for brevity
    chunk_ids = [f"{c.get('resource_type', 'Note')}:{c.get('resource_id', 'Unknown')}" for c in chunks]
    
    row = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "mode": mode,
        "retrieved_chunk_ids": "|".join(chunk_ids),
        "generated_text_length": len(generated_text),
        "unsupported_numbers": ",".join(map(str, unsupported_numbers)) if unsupported_numbers else "None",
        "unsupported_claims_count": len(unsupported_claims),
        "numeric_flag": len(unsupported_numbers) > 0,
        "semantic_flag": len(unsupported_claims) > 0,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
    }
    
    fieldnames = list(row.keys())
    
    with open(log_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
