"""Tests for semantic verification, provenance logging, and RAG integration."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import generate_mock_data
import rag_controller
import verifier

from tests.test_numeric_verifier import build_patient_context


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


class SemanticVerifierTests(unittest.TestCase):
    """Covers semantic claim handling and structured findings."""

    def setUp(self) -> None:
        self.verifier = verifier.SemanticVerifier()
        self.verifier.model = None
        self.verifier.reranker = None

    def test_splits_claims_and_ignores_headers(self) -> None:
        text = (
            "## Key Findings\n"
            "Chest pain is present.\n"
            "Questions to Ask Your Doctor\n"
            "Discuss this with your doctor.\n"
            "LDL cholesterol is elevated."
        )

        claims = self.verifier._split_into_claims(text)

        self.assertEqual(claims, ["Chest pain is present.", "LDL cholesterol is elevated."])

    def test_collects_sentence_level_evidence(self) -> None:
        chunks = [
            {
                "note_id": "note_a",
                "resource_type": "DocumentReference",
                "resource_id": "doc_a",
                "note_text": "Patient reports chest pain. Troponin is elevated.",
            }
        ]

        evidence_items = self.verifier._collect_evidence_sentences(chunks, build_patient_context("p001"))

        self.assertGreaterEqual(len(evidence_items), 3)
        self.assertEqual(evidence_items[0].sentence_text, "Patient reports chest pain.")
        self.assertEqual(evidence_items[1].sentence_text, "Troponin is elevated.")

    def test_supported_and_unsupported_claims(self) -> None:
        chunks = [
            {
                "note_id": "note_b",
                "resource_type": "DocumentReference",
                "resource_id": "doc_b",
                "note_text": "Patient reports chest pain. Troponin is elevated.",
            }
        ]

        supported = self.verifier.verify_detailed("Patient reports chest pain.", chunks)
        unsupported = self.verifier.verify_detailed("Patient has kidney failure.", chunks)

        self.assertEqual(supported, [])
        self.assertEqual(unsupported[0]["reason"], "anchor_mismatch")

    def test_negation_mismatch_is_flagged(self) -> None:
        chunks = [
            {
                "note_id": "note_c",
                "resource_type": "DocumentReference",
                "resource_id": "doc_c",
                "note_text": "Patient denies chest pain.",
            }
        ]

        findings = self.verifier.verify_detailed("Patient has chest pain.", chunks)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["reason"], "assertion_mismatch")

    def test_patient_context_supports_demographic_claim(self) -> None:
        findings = self.verifier.verify_detailed(
            generated_text="The patient is 67 years old.",
            retrieved_chunks=[],
            patient_context=build_patient_context("p001"),
        )

        self.assertEqual(findings, [])

    def test_numeric_only_mismatch_is_not_duplicated_semantically(self) -> None:
        chunks = [
            {
                "note_id": "note_d",
                "resource_type": "DocumentReference",
                "resource_id": "doc_d",
                "note_text": "LDL cholesterol was 142 mg/dL.",
            }
        ]

        numeric_findings = [
            {
                "claim_sentence": "LDL cholesterol was 150 mg/dL.",
                "claim_text": "150 mg/dL",
            }
        ]

        findings = self.verifier.verify_detailed(
            generated_text="LDL cholesterol was 150 mg/dL.",
            retrieved_chunks=chunks,
            numeric_findings=numeric_findings,
        )

        self.assertEqual(findings, [])


class SemanticBenchmarkTests(unittest.TestCase):
    """Ensures the semantic benchmark cases stay correctly classified."""

    def setUp(self) -> None:
        self.verifier = verifier.SemanticVerifier()
        self.verifier.model = None
        self.verifier.reranker = None
        self.notes = generate_mock_data.generate_note_records()

    def test_semantic_benchmark(self) -> None:
        with open(FIXTURE_DIR / "semantic_benchmark.json", encoding="utf-8") as fixture_file:
            cases = json.load(fixture_file)

        for case in cases:
            chunks = [self.notes[number - 1] for number in case["retrieved_note_numbers"]]
            patient_context = build_patient_context(case["patient_id"]) if case.get("patient_id") else None
            findings = self.verifier.verify_detailed(
                generated_text=case["generated_text"],
                retrieved_chunks=chunks,
                patient_context=patient_context,
                numeric_findings=case.get("numeric_findings"),
            )

            if case["expected_supported"]:
                self.assertEqual(findings, [], msg=case["name"])
            else:
                self.assertEqual(findings[0]["reason"], case["expected_reason"], msg=case["name"])


class ProvenanceLogTests(unittest.TestCase):
    """Validates the expanded audit CSV shape."""

    def test_log_provenance_stores_generated_text_and_serialized_findings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(verifier, "DATA_DIR", temp_dir):
                verifier.log_provenance(
                    query="lupus treatment plan",
                    mode="clinician",
                    chunks=[
                        {
                            "resource_type": "DocumentReference",
                            "resource_id": "doc_001",
                        }
                    ],
                    generated_text="Prednisone 40 mg daily was started.",
                    unsupported_numbers=["150"],
                    unsupported_claims=["Patient has chest pain."],
                    retrieval_ms=123,
                    generation_ms=456,
                    numeric_findings=[{"claim_text": "150 mg/dL", "reason": "out_of_tolerance"}],
                    semantic_findings=[{"claim_text": "Patient has chest pain.", "reason": "assertion_mismatch"}],
                )

                log_path = Path(temp_dir) / verifier.AUDIT_LOG_FILE
                with open(log_path, encoding="utf-8") as csvfile:
                    rows = list(csv.DictReader(csvfile))

                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["generated_text"], "Prednisone 40 mg daily was started.")
                self.assertEqual(
                    json.loads(rows[0]["numeric_findings_json"])[0]["reason"],
                    "out_of_tolerance",
                )
                self.assertEqual(
                    json.loads(rows[0]["semantic_findings_json"])[0]["reason"],
                    "assertion_mismatch",
                )


class RAGControllerVerificationIntegrationTests(unittest.TestCase):
    """Checks that controller results expose both verifier layers together."""

    def test_query_returns_numeric_and_semantic_findings(self) -> None:
        controller = rag_controller.RAGController.__new__(rag_controller.RAGController)

        class FakeRetriever:
            def retrieve(self, query: str, top_k: int = 8) -> list[dict]:
                return [
                    {
                        "rank": 1,
                        "note_id": "mock_note_001",
                        "patient_id": "p001",
                        "resource_type": "DocumentReference",
                        "resource_id": "doc_001",
                        "note_text": "LDL cholesterol was 142 mg/dL. Patient denies chest pain.",
                        "fused_score": 0.98,
                    }
                ]

        class FakeLLM:
            def generate(self, **kwargs) -> dict:
                return {
                    "response": "LDL cholesterol was 150 mg/dL. Patient has chest pain.",
                    "duration_ms": 42,
                }

        controller.retriever = FakeRetriever()
        controller.llm = FakeLLM()
        controller.numeric_verifier = verifier.NumericVerifier()
        controller.semantic_verifier = verifier.SemanticVerifier()
        controller.semantic_verifier.model = None
        controller.semantic_verifier.reranker = None
        controller._load_patient_context = lambda patient_id: build_patient_context(patient_id)

        result = controller.query(
            query_text="patient with chest pain and cholesterol issue",
            mode="clinician",
            top_k=1,
        )

        flags = result["verification_flags"]["clinician_response"]
        self.assertIn("numeric_findings", flags)
        self.assertIn("semantic_findings", flags)
        self.assertEqual(flags["numeric_findings"][0]["reason"], "out_of_tolerance")
        self.assertEqual(flags["semantic_findings"][0]["reason"], "assertion_mismatch")


if __name__ == "__main__":
    unittest.main()
