"""Tests for hybrid retrieval patient filtering."""

import unittest

import pandas as pd

import hybrid_retriever


class FakeDenseRetriever:
    """Deterministic dense retriever for unit tests."""

    def search(self, query: str, top_k: int = hybrid_retriever.DEFAULT_TOP_K) -> list[dict]:
        del query
        scores = [0.1, 0.2, 0.3, 0.4]
        return [
            {"index": index, "score": score}
            for index, score in enumerate(scores[:top_k])
        ]


class FakeBM25Retriever:
    """Deterministic lexical retriever for unit tests."""

    def search(self, query: str, top_k: int = hybrid_retriever.DEFAULT_TOP_K) -> list[dict]:
        del query
        scores = [4.0, 3.0, 2.0, 1.0]
        return [
            {"index": index, "score": score}
            for index, score in enumerate(scores[:top_k])
        ]


def build_retriever() -> hybrid_retriever.HybridRetriever:
    """Build a HybridRetriever without loading FAISS or BM25 dependencies."""
    retriever = hybrid_retriever.HybridRetriever.__new__(hybrid_retriever.HybridRetriever)
    retriever.metadata_df = pd.DataFrame([
        {
            "note_id": "note_1",
            "patient_id": "p001",
            "note_text": "Chest pain and diabetes.",
            "resource_type": "DocumentReference",
            "resource_id": "doc_1",
        },
        {
            "note_id": "note_2",
            "patient_id": "p002",
            "note_text": "Pneumonia follow up.",
            "resource_type": "DocumentReference",
            "resource_id": "doc_2",
        },
        {
            "note_id": "note_3",
            "patient_id": "p001",
            "note_text": "LDL cholesterol elevated.",
            "resource_type": "DocumentReference",
            "resource_id": "doc_3",
        },
        {
            "note_id": "note_4",
            "patient_id": "p003",
            "note_text": "Memory loss evaluation.",
            "resource_type": "DocumentReference",
            "resource_id": "doc_4",
        },
    ])
    retriever.dense_retriever = FakeDenseRetriever()
    retriever.bm25_retriever = FakeBM25Retriever()
    return retriever


class HybridPatientFilteringTests(unittest.TestCase):
    """Covers patient-scoped retrieval behavior."""

    def test_no_patient_id_preserves_unfiltered_results(self) -> None:
        retriever = build_retriever()

        results = retriever.retrieve(query="clinical summary", top_k=3)

        self.assertEqual(len(results), 3)
        self.assertGreater(len({result["patient_id"] for result in results}), 1)

    def test_patient_id_returns_only_matching_chunks(self) -> None:
        retriever = build_retriever()

        results = retriever.retrieve(query="clinical summary", top_k=3, patient_id="p001")

        self.assertEqual(len(results), 2)
        self.assertEqual({result["patient_id"] for result in results}, {"p001"})

    def test_empty_patient_match_returns_empty_list(self) -> None:
        retriever = build_retriever()

        results = retriever.retrieve(query="clinical summary", top_k=3, patient_id="missing")

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
