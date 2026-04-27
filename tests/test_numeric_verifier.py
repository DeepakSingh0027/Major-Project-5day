"""Tests for the Day 2 numeric verification pipeline."""

import json
import unittest
from pathlib import Path

import generate_mock_data

from verifier import NumericVerifier


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def build_patient_context(patient_id: str) -> dict | None:
    """Build patient context in the same shape used by the RAG controller."""
    patient_row = next(
        (row for row in generate_mock_data.MOCK_PATIENTS if row["patient_id"] == patient_id),
        None,
    )
    if patient_row is None:
        return None

    conditions = [
        row["condition"]
        for row in generate_mock_data.MOCK_CONDITIONS
        if row["patient_id"] == patient_id and row.get("clinical_status", "").lower() == "active"
    ]
    medications = [
        row["medication"]
        for row in generate_mock_data.MOCK_MEDICATIONS
        if row["patient_id"] == patient_id and row.get("status", "").lower() == "active"
    ]

    return {
        "patient_id": patient_id,
        "first_name": patient_row.get("first_name", ""),
        "last_name": patient_row.get("last_name", ""),
        "age": patient_row.get("age", ""),
        "gender": patient_row.get("gender", ""),
        "conditions": conditions,
        "medications": medications,
    }


class NumericExtractionTests(unittest.TestCase):
    """Covers structured mention extraction and ignore rules."""

    def setUp(self) -> None:
        self.verifier = NumericVerifier()

    def test_extracts_decimals_percentages_comma_numbers_doses_durations_and_slash_values(self) -> None:
        text = (
            "Platelets were 142,000. HbA1c was 8.2%. Blood pressure was 158/92 mmHg. "
            "Metformin 1000mg daily for 2 weeks. The patient is 67-year-old."
        )

        mentions = self.verifier.extract_mentions(text, source_kind="generated", source_id="sample")
        extracted = {(mention.raw_text, mention.unit, mention.category) for mention in mentions}

        self.assertIn(("142,000", "", "count"), extracted)
        self.assertIn(("8.2%", "%", "measurement"), extracted)
        self.assertIn(("158/92 mmHg", "mmhg", "blood_pressure"), extracted)
        self.assertIn(("1000mg", "mg", "dose"), extracted)
        self.assertIn(("2 weeks", "week", "duration"), extracted)
        self.assertIn(("67-year-old", "year", "age"), extracted)

    def test_ignores_dates_list_numbers_note_ranks_and_chunk_metadata(self) -> None:
        text = (
            "1. This is a numbered list.\n"
            "[2] ranked chunk\n"
            "Date 2025-04-01 noted.\n"
            "Note 3 referenced.\n"
            "mock_note_001 should be ignored.\n"
            "Blood pressure was 158/92 mmHg."
        )

        mentions = self.verifier.extract_mentions(text, source_kind="generated", source_id="sample")
        raw_mentions = [mention.raw_text for mention in mentions]

        self.assertEqual(raw_mentions, ["158/92 mmHg"])


class NumericMatchingTests(unittest.TestCase):
    """Covers matching policy and unsupported-claim reasons."""

    def setUp(self) -> None:
        self.verifier = NumericVerifier()

    def test_measurement_supports_exact_and_tolerance_matches(self) -> None:
        chunks = [
            {
                "note_id": "note_a",
                "resource_type": "Observation",
                "resource_id": "obs_a",
                "note_text": "Troponin I elevated at 0.08 ng/mL. LDL cholesterol 142 mg/dL.",
            }
        ]

        exact = self.verifier.verify_detailed("Troponin I was 0.08 ng/mL.", chunks)
        rounded = self.verifier.verify_detailed("LDL cholesterol was 143 mg/dL.", chunks)

        self.assertEqual(exact, [])
        self.assertEqual(rounded, [])

    def test_out_of_tolerance_and_unit_mismatch_are_flagged(self) -> None:
        chunks = [
            {
                "note_id": "note_b",
                "resource_type": "Observation",
                "resource_id": "obs_b",
                "note_text": "Creatinine 1.3 mg/dL. Troponin I elevated at 0.08 ng/mL.",
            }
        ]

        unit_mismatch = self.verifier.verify_detailed("Creatinine was 1.3 mmol/L.", chunks)
        out_of_tolerance = self.verifier.verify_detailed("Troponin I was 0.12 ng/mL.", chunks)

        self.assertEqual(unit_mismatch[0]["reason"], "unit_mismatch")
        self.assertEqual(out_of_tolerance[0]["reason"], "out_of_tolerance")

    def test_same_number_in_unrelated_context_is_flagged(self) -> None:
        chunks = [
            {
                "note_id": "note_c",
                "resource_type": "Encounter",
                "resource_id": "enc_c",
                "note_text": "Completed 4 cycles of treatment with good response.",
            }
        ]

        findings = self.verifier.verify_detailed("The plan includes 4 visits.", chunks)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["reason"], "context_mismatch")

    def test_age_can_be_supported_by_patient_context(self) -> None:
        patient_context = build_patient_context("p001")
        findings = self.verifier.verify_detailed(
            generated_text="The patient is 67 years old.",
            retrieved_chunks=[],
            patient_context=patient_context,
        )

        self.assertEqual(findings, [])

    def test_slash_component_fallback_matches_same_sentence_components(self) -> None:
        chunks = [
            {
                "note_id": "note_d",
                "resource_type": "Observation",
                "resource_id": "obs_d",
                "note_text": "Blood pressure today showed systolic 158 mmHg and diastolic 92 mmHg.",
            }
        ]

        findings = self.verifier.verify_detailed("Blood pressure was 158/92 mmHg.", chunks)

        self.assertEqual(findings, [])


class NumericIntegrationTests(unittest.TestCase):
    """Runs verifier checks against mock note records and canned outputs."""

    def setUp(self) -> None:
        self.verifier = NumericVerifier()
        self.notes = generate_mock_data.generate_note_records()

    def _chunks(self, *note_numbers: int) -> list[dict]:
        return [self.notes[number - 1] for number in note_numbers]

    def test_mixed_supported_and_unsupported_clinician_output(self) -> None:
        chunks = self._chunks(1, 2)
        patient_context = build_patient_context("p001")
        response = (
            "HbA1c is 8.2%. Blood pressure was 158/92 mmHg. "
            "The patient is 67 years old. LDL cholesterol was 150 mg/dL."
        )

        findings = self.verifier.verify_detailed(response, chunks, patient_context=patient_context)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["claim_text"], "150 mg/dL")
        self.assertEqual(findings[0]["reason"], "out_of_tolerance")

    def test_patient_friendly_output_supports_context_and_doses(self) -> None:
        chunks = self._chunks(9, 10)
        patient_context = build_patient_context("p005")
        response = (
            "Your Complement C3 was 65 mg/dL. "
            "Prednisone 40 mg daily was started. "
            "Labs should be checked every 2 weeks initially."
        )

        findings = self.verifier.verify_detailed(response, chunks, patient_context=patient_context)

        self.assertEqual(findings, [])


class NumericBenchmarkTests(unittest.TestCase):
    """Ensures the labeled benchmark stays above the target accuracy."""

    def setUp(self) -> None:
        self.verifier = NumericVerifier()
        self.notes = generate_mock_data.generate_note_records()

    def test_numeric_support_benchmark_meets_target(self) -> None:
        with open(FIXTURE_DIR / "numeric_benchmark.json", encoding="utf-8") as fixture_file:
            benchmark_cases = json.load(fixture_file)

        correct_predictions = 0
        for case in benchmark_cases:
            chunks = [self.notes[number - 1] for number in case["retrieved_note_numbers"]]
            patient_context = None
            if case.get("patient_id"):
                patient_context = build_patient_context(case["patient_id"])

            findings = self.verifier.verify_detailed(
                generated_text=case["generated_text"],
                retrieved_chunks=chunks,
                patient_context=patient_context,
            )

            predicted_supported = len(findings) == 0
            if predicted_supported == case["expected_supported"]:
                correct_predictions += 1

        accuracy = correct_predictions / len(benchmark_cases)
        self.assertGreaterEqual(accuracy, 0.96)


if __name__ == "__main__":
    unittest.main()
