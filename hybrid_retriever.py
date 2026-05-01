"""Hybrid retriever combining FAISS dense vectors with BM25 lexical search.

Fuses dense (semantic) and sparse (lexical) retrieval scores using
Reciprocal Rank Fusion to return the most relevant clinical note chunks
for a given query.
"""

import argparse
import math
import os
import re
import sys

import numpy as np
import pandas as pd

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    import faiss
except ImportError:
    faiss = None

try:
    import sentence_transformers
except ImportError:
    sentence_transformers = None

import vector_search


# -----------------
# Configuration
# -----------------
DATA_DIR = "datasets"
DEFAULT_DENSE_WEIGHT = 0.6
DEFAULT_LEXICAL_WEIGHT = 0.4
DEFAULT_TOP_K = 8
RRF_K = 60  # Reciprocal Rank Fusion constant


# -----------------
# BM25 Lexical Retriever
# -----------------
class BM25Retriever:
    """Lexical retriever using the BM25 (Okapi) scoring algorithm.

    Tokenizes note text into lowercased word tokens and builds a BM25
    index for fast keyword-based matching.
    """

    def __init__(self, documents: list[str]) -> None:
        if BM25Okapi is None:
            print("Error: 'rank_bm25' package is required for lexical search.")
            print("Install with: pip install rank-bm25")
            sys.exit(1)

        self.documents = documents
        tokenized = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace tokenizer with lowercasing and punctuation removal."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """Score all documents against a query and return top-k.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: index, score.
        """
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Rank by score, descending
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            if scores[idx] > 0:
                results.append({
                    "index": int(idx),
                    "score": float(scores[idx]),
                })

        return results


# -----------------
# Dense Retriever (FAISS wrapper)
# -----------------
class DenseRetriever:
    """Dense retriever using the existing FAISS index from vector_search.py.

    Loads pre-built vector artifacts and runs nearest-neighbor queries.
    """

    def __init__(self) -> None:
        if faiss is None:
            print("Error: 'faiss-cpu' package is required for dense search.")
            print("Install with: pip install faiss-cpu")
            sys.exit(1)

        if sentence_transformers is None:
            print("Error: 'sentence-transformers' package is required.")
            print("Install with: pip install sentence-transformers")
            sys.exit(1)

        if not vector_search.vector_artifacts_exist():
            print("Error: FAISS vector artifacts not found.")
            print("Run 'python vector_search.py' or 'python generate_mock_data.py' first.")
            sys.exit(1)

        self.metadata_df, self.index, self.config = vector_search.load_vector_artifacts()
        self.model = vector_search.load_embedding_model(self.config["model_name"])

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """Run a dense semantic search against the FAISS index.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: index, score (L2 distance).
        """
        query_embedding = vector_search.encode_texts(self.model, [query])
        actual_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, actual_k)

        results = []
        for distance, row_index in zip(distances[0], indices[0]):
            if row_index < 0:
                continue
            results.append({
                "index": int(row_index),
                "score": float(distance),
            })

        return results


# -----------------
# Score Normalization
# -----------------
def normalize_scores_min_max(scores: list[float], invert: bool = False) -> list[float]:
    """Normalize scores to [0, 1] range using min-max scaling.

    Args:
        scores: Raw scores.
        invert: If True, invert so lower raw scores → higher normalized
                scores (used for L2 distances).

    Returns:
        List of normalized scores in [0, 1].
    """
    if not scores:
        return []

    min_s = min(scores)
    max_s = max(scores)
    spread = max_s - min_s

    if spread == 0:
        return [1.0] * len(scores)

    if invert:
        return [(max_s - s) / spread for s in scores]
    return [(s - min_s) / spread for s in scores]


# -----------------
# Hybrid Retriever
# -----------------
class HybridRetriever:
    """Fuses dense (FAISS) and lexical (BM25) retrieval using RRF.

    Combines semantic understanding from sentence embeddings with exact
    keyword matching from BM25 to produce a unified ranked list.
    """

    def __init__(self, metadata_df: pd.DataFrame | None = None) -> None:
        self.dense_retriever = DenseRetriever()

        # Use provided metadata or fall back to dense retriever's metadata
        if metadata_df is not None:
            self.metadata_df = metadata_df
        else:
            self.metadata_df = self.dense_retriever.metadata_df

        # Build BM25 index over the same note corpus
        note_texts = self.metadata_df["note_text"].fillna("").tolist()
        self.bm25_retriever = BM25Retriever(note_texts)

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
        lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
        patient_id: str | None = None,
    ) -> list[dict]:
        """Run hybrid retrieval and return fused results.

        Uses Reciprocal Rank Fusion (RRF) to combine dense and lexical
        rankings into a single scored list.

        Args:
            query: Natural-language search query.
            top_k: Number of final results to return (target 5-8).
            dense_weight: Weight for dense retrieval scores in fusion.
            lexical_weight: Weight for lexical retrieval scores in fusion.
            patient_id: Optional patient ID used to filter retrieved chunks.

        Returns:
            List of dicts, each containing: note_id, patient_id,
            note_text, dense_score, lexical_score, fused_score, rank.
        """
        # FAISS cannot filter metadata natively, so patient-scoped retrieval
        # over-fetches and filters after vector/BM25 scoring.
        candidate_multiplier = 10 if patient_id else 3
        candidate_k = min(top_k * candidate_multiplier, len(self.metadata_df))

        # 1. Dense retrieval
        dense_results = self.dense_retriever.search(query, candidate_k)

        # 2. Lexical retrieval
        lexical_results = self.bm25_retriever.search(query, candidate_k)

        # 3. Build per-index score maps
        # Dense scores are L2 distances — lower is better → invert
        dense_scores_raw = {r["index"]: r["score"] for r in dense_results}
        lexical_scores_raw = {r["index"]: r["score"] for r in lexical_results}

        # Normalize dense scores (invert because L2 distance)
        if dense_scores_raw:
            dense_indices = list(dense_scores_raw.keys())
            dense_vals = [dense_scores_raw[i] for i in dense_indices]
            dense_norm = normalize_scores_min_max(dense_vals, invert=True)
            dense_normalized = dict(zip(dense_indices, dense_norm))
        else:
            dense_normalized = {}

        # Normalize lexical scores (higher is better)
        if lexical_scores_raw:
            lex_indices = list(lexical_scores_raw.keys())
            lex_vals = [lexical_scores_raw[i] for i in lex_indices]
            lex_norm = normalize_scores_min_max(lex_vals, invert=False)
            lexical_normalized = dict(zip(lex_indices, lex_norm))
        else:
            lexical_normalized = {}

        # 4. Reciprocal Rank Fusion
        all_candidate_indices = set(dense_normalized.keys()) | set(lexical_normalized.keys())

        fused_scores = {}
        for idx in all_candidate_indices:
            d_score = dense_normalized.get(idx, 0.0)
            l_score = lexical_normalized.get(idx, 0.0)
            fused = dense_weight * d_score + lexical_weight * l_score
            fused_scores[idx] = {
                "dense_score": round(d_score, 4),
                "lexical_score": round(l_score, 4),
                "fused_score": round(fused, 4),
            }

        # 5. Sort by fused score and take top-k after optional patient filter
        ranked_indices = sorted(
            fused_scores.keys(),
            key=lambda i: fused_scores[i]["fused_score"],
            reverse=True,
        )
        if patient_id:
            normalized_patient_id = str(patient_id)
            ranked_indices = [
                index
                for index in ranked_indices
                if str(self.metadata_df.iloc[index].get("patient_id", "")) == normalized_patient_id
            ]
        ranked_indices = ranked_indices[:top_k]

        # 6. Build result dicts with metadata
        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            row = self.metadata_df.iloc[idx]
            result = {
                "rank": rank,
                "note_id": str(row.get("note_id", "")),
                "patient_id": str(row.get("patient_id", "")),
                "note_text": str(row.get("note_text", "")),
                "resource_type": str(row.get("resource_type", "")),
                "resource_id": str(row.get("resource_id", "")),
                "date": str(row.get("date", "")),
                "dense_score": fused_scores[idx]["dense_score"],
                "lexical_score": fused_scores[idx]["lexical_score"],
                "fused_score": fused_scores[idx]["fused_score"],
            }
            results.append(result)

        return results


def print_retrieval_results(results: list[dict], query: str) -> None:
    """Print hybrid retrieval results in a readable terminal format."""
    if not results:
        print(f"No results found for query: '{query}'")
        return

    print(f"\nHybrid retrieval results for: '{query}'")
    print(f"Retrieved {len(results)} chunk(s)\n")

    for result in results:
        note_preview = result["note_text"].replace("\n", " ").strip()
        if len(note_preview) > 150:
            note_preview = f"{note_preview[:147]}..."

        print(
            f"[{result['rank']}] patient={result['patient_id']} "
            f"fused={result['fused_score']:.4f} "
            f"(dense={result['dense_score']:.4f}, lex={result['lexical_score']:.4f})"
        )
        print(f"    note_id={result['note_id']}")
        print(f"    {note_preview}\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone hybrid retrieval."""
    parser = argparse.ArgumentParser(
        description="Run hybrid (dense + BM25) retrieval over clinical notes."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="patients with chest pain and diabetes",
        help="Search query string.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of results to return.",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=DEFAULT_DENSE_WEIGHT,
        help="Weight for dense retrieval scores.",
    )
    parser.add_argument(
        "--lexical-weight",
        type=float,
        default=DEFAULT_LEXICAL_WEIGHT,
        help="Weight for BM25 lexical retrieval scores.",
    )
    return parser.parse_args()


def main() -> None:
    """Build hybrid retriever and run a test query."""
    args = parse_args()

    print("Initializing hybrid retriever...")
    retriever = HybridRetriever()
    print(f"Loaded {len(retriever.metadata_df)} note(s) into hybrid index.\n")

    results = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
        dense_weight=args.dense_weight,
        lexical_weight=args.lexical_weight,
    )

    print_retrieval_results(results, args.query)


if __name__ == "__main__":
    main()
