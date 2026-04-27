"""Verifier and audit module for the RAG pipeline.

Implements safety layers to check LLM outputs against source evidence
and logs all executions to an audit trail.
"""

import csv
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer, util
except ImportError:
    CrossEncoder = None
    SentenceTransformer = None
    util = None


# -----------------
# Configuration
# -----------------
DATA_DIR = "datasets"
AUDIT_LOG_FILE = "audit_log.csv"
SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"
SEMANTIC_RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SEMANTIC_THRESHOLD = 0.5
SEMANTIC_BI_ENCODER_THRESHOLD = 0.35
SEMANTIC_RERANKER_THRESHOLD = 0.32
SEMANTIC_SHORTLIST_K = 5

DEFAULT_RELATIVE_TOLERANCE = 0.05
DEFAULT_ABSOLUTE_DECIMAL_TOLERANCE = 0.02
DEFAULT_INTEGER_TOLERANCE = 1.0
CONTEXT_MATCH_THRESHOLD = 0.2

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "if",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "patient",
    "she",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "were",
    "with",
    "your",
}

SEMANTIC_GENERIC_TOKENS = {
    "doctor",
    "discuss",
    "follow",
    "followup",
    "finding",
    "findings",
    "health",
    "monitor",
    "plan",
    "provider",
    "question",
    "questions",
    "result",
    "results",
    "review",
    "summary",
    "watch",
}

DURATION_UNITS = {
    "day",
    "week",
    "month",
    "year",
    "hour",
    "minute",
}

DOSE_UNITS = {
    "mg",
    "mcg",
    "g",
    "kg",
    "ml",
    "l",
    "unit",
    "units",
    "iu",
}

MEASUREMENT_UNITS = {
    "%",
    "mmhg",
    "bpm",
    "mg/dl",
    "ng/ml",
    "iu/ml",
    "u/l",
    "meq/l",
    "ml/min",
    "ml/hr",
    "g",
    "g/dl",
    "kg/m2",
    "f",
}

GENERAL_NUMBER_RE = re.compile(
    r"(?<![\w/.-])"
    r"(?P<number>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    r"(?P<unit>\s*(?:%|[A-Za-z][A-Za-z0-9/%-]*))?"
    r"(?![\w/-])"
)
SLASH_NUMBER_RE = re.compile(
    r"(?<![\w.-])"
    r"(?P<left>\d{1,3}(?:,\d{3})*(?:\.\d+)?)"
    r"/"
    r"(?P<right>\d{1,3}(?:,\d{3})*(?:\.\d+)?)"
    r"(?P<unit>\s*(?:%|[A-Za-z][A-Za-z0-9/%-]*))?"
    r"(?![\w/-])"
)
AGE_RE = re.compile(r"(?<!\w)(?P<age>\d{1,3})\s*-\s*year-old(?!\w)", re.IGNORECASE)
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
TIMESTAMP_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?\b")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")


@dataclass
class NumericMention:
    """Structured numeric mention used for source and claim matching."""

    raw_text: str
    value: float | tuple[float, float]
    unit: str
    category: str
    context_text: str
    sentence_text: str
    source_kind: str
    source_id: str
    start_char: int
    end_char: int

    @property
    def is_ratio(self) -> bool:
        """Return True when the mention stores a slash-style numeric value."""
        return isinstance(self.value, tuple)

    @property
    def normalized_value(self) -> float | str:
        """Return a JSON-friendly normalized value."""
        if self.is_ratio:
            left, right = self.value
            return f"{_format_numeric(left)}/{_format_numeric(right)}"
        return float(self.value)


@dataclass
class SemanticEvidence:
    """Sentence-level evidence candidate for semantic verification."""

    sentence_text: str
    source_kind: str
    source_id: str
    source_text: str


def _format_numeric(value: float) -> str:
    """Format numeric values without trailing decimal noise."""
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):g}"


def _normalize_number(number_text: str) -> float:
    """Parse a numeric string after stripping comma separators."""
    return float(number_text.replace(",", ""))


def _normalize_unit(unit_text: str) -> str:
    """Normalize units into a compact matching form."""
    if not unit_text:
        return ""

    unit = unit_text.strip().lower().rstrip(".,;:)")
    unit = unit.replace("(", "")
    unit = re.sub(r"\s+", "", unit)

    aliases = {
        "yrs": "year",
        "yr": "year",
        "years": "year",
        "year": "year",
        "days": "day",
        "day": "day",
        "weeks": "week",
        "week": "week",
        "months": "month",
        "month": "month",
        "hours": "hour",
        "hour": "hour",
        "mins": "minute",
        "min": "minute",
        "minutes": "minute",
        "minute": "minute",
        "mmhg": "mmhg",
        "mg/dl": "mg/dl",
        "ng/ml": "ng/ml",
        "iu/ml": "iu/ml",
        "u/l": "u/l",
        "meq/l": "meq/l",
        "ml/min": "ml/min",
        "ml/hr": "ml/hr",
        "bpm": "bpm",
        "kg/m2": "kg/m2",
        "°f": "f",
    }

    return aliases.get(unit, unit)


def _split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Split text into sentence-like spans while keeping offsets."""
    spans = []
    start = 0

    for index, character in enumerate(text):
        if character not in ".!?\n":
            continue

        previous_char = text[index - 1] if index > 0 else ""
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if character == "." and previous_char.isdigit() and next_char.isdigit():
            continue

        raw_sentence = text[start:index + 1]
        stripped_sentence = raw_sentence.strip()
        if not stripped_sentence:
            start = index + 1
            continue

        leading = len(raw_sentence) - len(raw_sentence.lstrip())
        trailing = len(raw_sentence) - len(raw_sentence.rstrip())
        sentence_start = start + leading
        sentence_end = index + 1 - trailing
        spans.append((sentence_start, sentence_end, stripped_sentence))
        start = index + 1

    if start < len(text):
        raw_sentence = text[start:]
        stripped_sentence = raw_sentence.strip()
        if stripped_sentence:
            leading = len(raw_sentence) - len(raw_sentence.lstrip())
            trailing = len(raw_sentence) - len(raw_sentence.rstrip())
            sentence_start = start + leading
            sentence_end = len(text) - trailing
            spans.append((sentence_start, sentence_end, stripped_sentence))

    return spans


def _find_ignored_spans(text: str) -> list[tuple[int, int]]:
    """Return spans that should be excluded from numeric extraction."""
    ignored_spans = []
    for pattern in (TIMESTAMP_RE, DATE_RE, TIME_RE):
        for match in pattern.finditer(text):
            ignored_spans.append((match.start(), match.end()))
    return ignored_spans


def _spans_overlap(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    """Return True if a span overlaps any span in the provided list."""
    for span_start, span_end in spans:
        if start < span_end and end > span_start:
            return True
    return False


def _extract_context_snippet(sentence: str, start: int, end: int, window: int = 48) -> str:
    """Extract a local context window around a numeric mention."""
    snippet_start = max(0, start - window)
    snippet_end = min(len(sentence), end + window)
    return sentence[snippet_start:snippet_end].strip()


def _tokenize_context(text: str) -> set[str]:
    """Tokenize context text for lightweight overlap scoring."""
    tokens = set()
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9+-]*", text.lower()):
        if token in STOPWORDS:
            continue
        if _normalize_unit(token) in DURATION_UNITS | DOSE_UNITS | MEASUREMENT_UNITS:
            continue
        tokens.add(token)
    return tokens


def _render_patient_context(patient_context: dict | None) -> str:
    """Render patient context into prompt-like lines for reuse by verifiers."""
    if not patient_context:
        return ""

    lines = []
    if patient_context.get("patient_id"):
        lines.append(f"Patient ID: {patient_context['patient_id']}")
    if patient_context.get("first_name") or patient_context.get("last_name"):
        full_name = (
            f"{patient_context.get('first_name', '')} "
            f"{patient_context.get('last_name', '')}"
        ).strip()
        lines.append(f"Name: {full_name}")
    if patient_context.get("age") not in (None, ""):
        lines.append(f"The patient is {patient_context['age']} years old.")
    if patient_context.get("gender"):
        lines.append(f"Gender: {patient_context['gender']}")

    conditions = patient_context.get("conditions", [])
    if conditions:
        lines.append(f"Active conditions: {', '.join(conditions)}")

    medications = patient_context.get("medications", [])
    if medications:
        lines.append(f"Current medications: {', '.join(medications)}")

    return "\n".join(lines)


def _safe_float(value: float | int | None) -> float | None:
    """Return a rounded float for stable serialization."""
    if value is None:
        return None
    return round(float(value), 6)


def _json_dumps_compact(payload: object) -> str:
    """Serialize complex payloads into deterministic compact JSON."""
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


class NumericVerifier:
    """Verifies generated numeric claims against retrieved evidence."""

    def __init__(
        self,
        tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
        absolute_decimal_tolerance: float = DEFAULT_ABSOLUTE_DECIMAL_TOLERANCE,
        integer_tolerance: float = DEFAULT_INTEGER_TOLERANCE,
    ):
        """Initialize the numeric verifier and tolerance policy."""
        self.tolerance = tolerance
        self.absolute_decimal_tolerance = absolute_decimal_tolerance
        self.integer_tolerance = integer_tolerance

    def extract_numbers(self, text: str) -> list[float]:
        """Backward-compatible numeric extraction helper."""
        values = []
        for mention in self.extract_mentions(text, source_kind="text", source_id="text"):
            if mention.is_ratio:
                values.extend([float(mention.value[0]), float(mention.value[1])])
            else:
                values.append(float(mention.value))
        return values

    def extract_mentions(
        self,
        text: str,
        source_kind: str,
        source_id: str,
    ) -> list[NumericMention]:
        """Extract structured numeric mentions from arbitrary text."""
        if not text.strip():
            return []

        mentions: list[NumericMention] = []
        ignored_spans = _find_ignored_spans(text)
        used_spans: list[tuple[int, int]] = []

        for sentence_start, _, sentence_text in _split_sentences(text):
            sentence_mentions = self._extract_sentence_mentions(
                sentence_text=sentence_text,
                sentence_start=sentence_start,
                source_kind=source_kind,
                source_id=source_id,
                ignored_spans=ignored_spans,
                used_spans=used_spans,
            )
            mentions.extend(sentence_mentions)

        return mentions

    def extract_patient_context_mentions(self, patient_context: dict | None) -> list[NumericMention]:
        """Extract numeric mentions from rendered patient context."""
        if not patient_context:
            return []
        patient_id = str(patient_context.get("patient_id", "patient_context"))
        rendered_context = _render_patient_context(patient_context)
        return self.extract_mentions(
            rendered_context,
            source_kind="patient_context",
            source_id=patient_id,
        )

    def verify_detailed(
        self,
        generated_text: str,
        retrieved_chunks: list[dict],
        patient_context: dict | None = None,
    ) -> list[dict]:
        """Return one finding per unsupported generated numeric claim."""
        claims = self.extract_mentions(
            generated_text,
            source_kind="generated",
            source_id="generated_response",
        )
        if not claims:
            return []

        evidence_mentions = self._collect_evidence_mentions(retrieved_chunks, patient_context)
        if not evidence_mentions:
            return [
                self._build_finding(
                    claim=claim,
                    reason="no_evidence_found",
                    best_candidate=None,
                    difference=None,
                )
                for claim in claims
            ]

        findings = []
        for claim in claims:
            is_supported, reason, best_candidate, difference = self._match_claim(claim, evidence_mentions)
            if not is_supported:
                findings.append(
                    self._build_finding(
                        claim=claim,
                        reason=reason,
                        best_candidate=best_candidate,
                        difference=difference,
                    )
                )

        return findings

    def verify(
        self,
        generated_text: str,
        retrieved_chunks: list[dict],
        patient_context: dict | None = None,
    ) -> list[float | str]:
        """Backward-compatible wrapper returning compact unsupported values."""
        findings = self.verify_detailed(
            generated_text=generated_text,
            retrieved_chunks=retrieved_chunks,
            patient_context=patient_context,
        )

        unsupported = []
        for finding in findings:
            value = finding["normalized_value"]
            unsupported.append(value)

        compact = []
        seen = set()
        for value in unsupported:
            marker = str(value)
            if marker in seen:
                continue
            seen.add(marker)
            compact.append(value)
        return compact

    def _extract_sentence_mentions(
        self,
        sentence_text: str,
        sentence_start: int,
        source_kind: str,
        source_id: str,
        ignored_spans: list[tuple[int, int]],
        used_spans: list[tuple[int, int]],
    ) -> list[NumericMention]:
        """Extract numeric mentions from a single sentence."""
        mentions = []

        for match in SLASH_NUMBER_RE.finditer(sentence_text):
            start, end = sentence_start + match.start(), sentence_start + match.end()
            if _spans_overlap(start, end, ignored_spans) or _spans_overlap(start, end, used_spans):
                continue
            if self._should_ignore_match(sentence_text, match.start(), match.end(), match.group(0)):
                continue
            if self._looks_like_slash_date(match.group(0)):
                continue

            unit = _normalize_unit(match.group("unit") or "")
            left = _normalize_number(match.group("left"))
            right = _normalize_number(match.group("right"))
            category = "blood_pressure" if unit == "mmhg" or "blood pressure" in sentence_text.lower() else "ratio"
            if category == "blood_pressure" and not unit:
                unit = "mmhg"

            mentions.append(
                NumericMention(
                    raw_text=match.group(0).strip(),
                    value=(left, right),
                    unit=unit,
                    category=category,
                    context_text=_extract_context_snippet(sentence_text, match.start(), match.end()),
                    sentence_text=sentence_text.strip(),
                    source_kind=source_kind,
                    source_id=source_id,
                    start_char=start,
                    end_char=end,
                )
            )
            used_spans.append((start, end))

        for match in AGE_RE.finditer(sentence_text):
            start, end = sentence_start + match.start(), sentence_start + match.end()
            if _spans_overlap(start, end, ignored_spans) or _spans_overlap(start, end, used_spans):
                continue

            age = _normalize_number(match.group("age"))
            mentions.append(
                NumericMention(
                    raw_text=match.group(0).strip(),
                    value=age,
                    unit="year",
                    category="age",
                    context_text=_extract_context_snippet(sentence_text, match.start(), match.end()),
                    sentence_text=sentence_text.strip(),
                    source_kind=source_kind,
                    source_id=source_id,
                    start_char=start,
                    end_char=end,
                )
            )
            used_spans.append((start, end))

        for match in GENERAL_NUMBER_RE.finditer(sentence_text):
            start, end = sentence_start + match.start(), sentence_start + match.end()
            if _spans_overlap(start, end, ignored_spans) or _spans_overlap(start, end, used_spans):
                continue
            if self._should_ignore_match(sentence_text, match.start(), match.end(), match.group(0)):
                continue

            raw_text = match.group(0).strip()
            value = _normalize_number(match.group("number"))
            unit = _normalize_unit(match.group("unit") or "")
            category = self._classify_general_mention(sentence_text, raw_text, unit)

            if category == "age" and not unit:
                unit = "year"

            mentions.append(
                NumericMention(
                    raw_text=raw_text,
                    value=value,
                    unit=unit,
                    category=category,
                    context_text=_extract_context_snippet(sentence_text, match.start(), match.end()),
                    sentence_text=sentence_text.strip(),
                    source_kind=source_kind,
                    source_id=source_id,
                    start_char=start,
                    end_char=end,
                )
            )
            used_spans.append((start, end))

        return mentions

    def _classify_general_mention(self, sentence_text: str, raw_text: str, unit: str) -> str:
        """Classify a general numeric mention into a verification category."""
        lower_sentence = sentence_text.lower()

        if unit == "year" and re.search(r"\byears?\s+old\b", lower_sentence):
            return "age"
        if unit == "year" and lower_sentence.startswith("age:"):
            return "age"
        if unit in DURATION_UNITS:
            return "duration"
        if unit in DOSE_UNITS:
            return "dose"
        if unit in MEASUREMENT_UNITS:
            return "measurement"
        if "/" in unit:
            return "measurement"
        if "%" in raw_text or unit == "%":
            return "measurement"
        if "age" in lower_sentence:
            return "age"
        if "day" in lower_sentence and raw_text.lower().startswith("day "):
            return "duration"

        return "count"

    def _looks_like_slash_date(self, raw_text: str) -> bool:
        """Return True for mm/dd-style patterns that should be ignored."""
        match = re.fullmatch(r"\s*(\d{1,2})/(\d{1,2})\s*", raw_text)
        if not match:
            return False

        left = int(match.group(1))
        right = int(match.group(2))
        return left <= 12 and right <= 31

    def _should_ignore_match(
        self,
        sentence_text: str,
        start: int,
        end: int,
        raw_text: str,
    ) -> bool:
        """Filter out numbering, chunk metadata, and prompt formatting noise."""
        stripped_before = sentence_text[:start].strip()
        after_text = sentence_text[end:]

        if not stripped_before and re.match(r"^[.)]\s*$", after_text.strip()):
            return True
        if not stripped_before and re.match(r"^[.)]\s+", after_text):
            return True
        if start > 0 and sentence_text[start - 1] == "[" and end < len(sentence_text) and sentence_text[end] == "]":
            return True
        if re.search(r"(note|chunk|rank|section)\s*$", sentence_text[:start].lower()):
            return True
        if raw_text.startswith("#"):
            return True

        return False

    def _collect_evidence_mentions(
        self,
        retrieved_chunks: list[dict],
        patient_context: dict | None,
    ) -> list[NumericMention]:
        """Collect numeric mentions from retrieved chunks and patient context."""
        mentions = []

        for chunk in retrieved_chunks:
            source_id_parts = [
                str(chunk.get("note_id", "")),
                str(chunk.get("resource_type", "")),
                str(chunk.get("resource_id", "")),
            ]
            source_id = ":".join([part for part in source_id_parts if part]) or "chunk"
            mentions.extend(
                self.extract_mentions(
                    str(chunk.get("note_text", "")),
                    source_kind="retrieved_chunk",
                    source_id=source_id,
                )
            )

        mentions.extend(self.extract_patient_context_mentions(patient_context))
        return mentions

    def _match_claim(
        self,
        claim: NumericMention,
        evidence_mentions: list[NumericMention],
    ) -> tuple[bool, str, NumericMention | None, float | str | None]:
        """Match one claim against evidence mentions."""
        compatible_candidates = []
        value_mismatch_candidates = []
        context_mismatch_candidates = []
        unit_mismatch_candidates = []

        for candidate in evidence_mentions:
            if not self._category_compatible(claim, candidate):
                continue

            context_score = self._context_overlap(claim, candidate)
            units_match = self._units_compatible(claim, candidate)

            if not units_match:
                unit_mismatch_candidates.append((candidate, context_score))
                continue

            compatible_candidates.append((candidate, context_score))
            values_match, difference = self._values_match(claim, candidate)
            if values_match:
                if self._requires_context_match(claim) and not self._context_is_strong_enough(
                    claim,
                    candidate,
                    context_score,
                ):
                    context_mismatch_candidates.append((candidate, context_score))
                    continue
                return True, "supported", candidate, difference

            value_mismatch_candidates.append((candidate, context_score))

        if claim.is_ratio:
            ratio_supported, ratio_candidate = self._ratio_component_fallback(claim, evidence_mentions)
            if ratio_supported:
                return True, "supported", ratio_candidate, 0.0

        if compatible_candidates:
            best_candidate, _ = self._select_best_candidate(
                claim,
                [candidate for candidate, _ in compatible_candidates],
            )
            if context_mismatch_candidates and not value_mismatch_candidates:
                return False, "context_mismatch", best_candidate, None

            difference = None
            if best_candidate is not None:
                _, difference = self._values_match(claim, best_candidate)
            return False, "out_of_tolerance", best_candidate, difference

        if unit_mismatch_candidates:
            best_candidate, _ = self._select_best_candidate(
                claim,
                [candidate for candidate, _ in unit_mismatch_candidates],
            )
            return False, "unit_mismatch", best_candidate, None

        best_candidate, _ = self._select_best_candidate(claim, evidence_mentions)
        return False, "no_evidence_found", best_candidate, None

    def _category_compatible(self, claim: NumericMention, candidate: NumericMention) -> bool:
        """Return True when claim and candidate can be compared."""
        if claim.category == candidate.category:
            return True
        if {claim.category, candidate.category} == {"blood_pressure", "ratio"}:
            return True
        return False

    def _units_compatible(self, claim: NumericMention, candidate: NumericMention) -> bool:
        """Return True when units are compatible for comparison."""
        if claim.category == "count" and candidate.category == "count":
            return True

        if claim.unit and candidate.unit:
            return claim.unit == candidate.unit

        if claim.category in {"age", "count"}:
            return True

        return claim.unit == candidate.unit

    def _requires_context_match(self, claim: NumericMention) -> bool:
        """Return True when a claim needs context overlap to count as supported."""
        return claim.category in {"count", "dose", "duration"}

    def _context_is_strong_enough(
        self,
        claim: NumericMention,
        candidate: NumericMention,
        score: float,
    ) -> bool:
        """Return True when context overlap is sufficient for support."""
        if claim.category == "age" and candidate.source_kind == "patient_context":
            return True
        return score >= CONTEXT_MATCH_THRESHOLD

    def _context_overlap(self, claim: NumericMention, candidate: NumericMention) -> float:
        """Compute simple token overlap between claim and candidate context."""
        claim_tokens = _tokenize_context(claim.sentence_text)
        candidate_tokens = _tokenize_context(candidate.sentence_text)

        if not claim_tokens or not candidate_tokens:
            return 0.0

        overlap = claim_tokens & candidate_tokens
        return len(overlap) / max(len(claim_tokens), 1)

    def _values_match(
        self,
        claim: NumericMention,
        candidate: NumericMention,
    ) -> tuple[bool, float | str]:
        """Return whether two numeric mentions match and their difference."""
        if claim.is_ratio and candidate.is_ratio:
            claim_left, claim_right = claim.value
            cand_left, cand_right = candidate.value

            left_match, left_diff = self._single_value_match(
                claim_left,
                cand_left,
                category="measurement",
            )
            right_match, right_diff = self._single_value_match(
                claim_right,
                cand_right,
                category="measurement",
            )

            if left_match and right_match:
                return True, 0.0

            difference = f"systolic={left_diff}, diastolic={right_diff}"
            return False, difference

        if claim.is_ratio or candidate.is_ratio:
            return False, None

        return self._single_value_match(
            float(claim.value),
            float(candidate.value),
            category=claim.category,
        )

    def _single_value_match(
        self,
        claim_value: float,
        candidate_value: float,
        category: str,
    ) -> tuple[bool, float]:
        """Apply the default tolerance policy for single numeric values."""
        difference = abs(claim_value - candidate_value)

        if category in {"age", "count", "dose", "duration"}:
            return difference == 0.0, difference

        if difference == 0.0:
            return True, 0.0

        if float(claim_value).is_integer() and float(candidate_value).is_integer():
            if difference <= self.integer_tolerance:
                return True, difference

        if abs(candidate_value) < 1.0 and difference <= self.absolute_decimal_tolerance:
            return True, difference

        denominator = abs(candidate_value) if candidate_value != 0 else 1.0
        relative_difference = difference / denominator
        return relative_difference <= self.tolerance, round(relative_difference, 6)

    def _ratio_component_fallback(
        self,
        claim: NumericMention,
        evidence_mentions: list[NumericMention],
    ) -> tuple[bool, NumericMention | None]:
        """Allow slash expressions to match component numbers from one sentence."""
        if not claim.is_ratio:
            return False, None

        sentence_groups: dict[tuple[str, str], list[NumericMention]] = {}
        for candidate in evidence_mentions:
            key = (candidate.source_id, candidate.sentence_text)
            sentence_groups.setdefault(key, []).append(candidate)

        claim_left, claim_right = claim.value
        for sentence_mentions in sentence_groups.values():
            left_supported = False
            right_supported = False
            representative = None

            for candidate in sentence_mentions:
                if candidate.is_ratio:
                    continue
                if claim.unit and candidate.unit and claim.unit != candidate.unit:
                    continue

                if representative is None:
                    representative = candidate

                left_match, _ = self._single_value_match(
                    claim_left,
                    float(candidate.value),
                    category="measurement",
                )
                right_match, _ = self._single_value_match(
                    claim_right,
                    float(candidate.value),
                    category="measurement",
                )

                left_supported = left_supported or left_match
                right_supported = right_supported or right_match

            if left_supported and right_supported:
                return True, representative

        return False, None

    def _select_best_candidate(
        self,
        claim: NumericMention,
        candidates: list[NumericMention],
    ) -> tuple[NumericMention | None, float]:
        """Pick the best candidate using context overlap and numeric closeness."""
        best_candidate = None
        best_score = -1.0

        for candidate in candidates:
            context_score = self._context_overlap(claim, candidate)
            closeness_bonus = 0.0

            if claim.is_ratio and candidate.is_ratio:
                claim_left, claim_right = claim.value
                cand_left, cand_right = candidate.value
                closeness_bonus = 1.0 / (
                    1.0 + abs(claim_left - cand_left) + abs(claim_right - cand_right)
                )
            elif not claim.is_ratio and not candidate.is_ratio:
                closeness_bonus = 1.0 / (1.0 + abs(float(claim.value) - float(candidate.value)))

            unit_bonus = 1.0 if self._units_compatible(claim, candidate) else 0.0
            score = (context_score * 3.0) + closeness_bonus + unit_bonus

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate, best_score

    def _build_finding(
        self,
        claim: NumericMention,
        reason: str,
        best_candidate: NumericMention | None,
        difference: float | str | None,
    ) -> dict:
        """Build the structured unsupported-claim finding."""
        return {
            "claim_text": claim.raw_text,
            "normalized_value": claim.normalized_value,
            "unit": claim.unit,
            "claim_sentence": claim.sentence_text,
            "reason": reason,
            "best_candidate_source": best_candidate.raw_text if best_candidate else "",
            "best_candidate_source_id": best_candidate.source_id if best_candidate else "",
            "difference": difference,
        }


class SemanticVerifier:
    """Uses a hybrid semantic verifier to check claim consistency."""

    def __init__(
        self,
        model_name: str = SEMANTIC_MODEL_NAME,
        threshold: float = SEMANTIC_THRESHOLD,
        reranker_model_name: str = SEMANTIC_RERANKER_MODEL_NAME,
        bi_encoder_threshold: float = SEMANTIC_BI_ENCODER_THRESHOLD,
        reranker_threshold: float = SEMANTIC_RERANKER_THRESHOLD,
        shortlist_k: int = SEMANTIC_SHORTLIST_K,
        embedding_model=None,
        reranker_model=None,
    ):
        """Initialize semantic models and thresholds."""
        self.threshold = threshold
        self.bi_encoder_threshold = bi_encoder_threshold
        self.reranker_threshold = reranker_threshold
        self.shortlist_k = shortlist_k

        if embedding_model is not None:
            self.model = embedding_model
        elif SentenceTransformer is not None:
            try:
                print(f"  Loading Semantic Verifier ({model_name})...")
                self.model = SentenceTransformer(model_name)
            except Exception as exc:
                print(f"  Warning: semantic bi-encoder unavailable ({exc}). Using lexical fallback.")
                self.model = None
        else:
            print("  Warning: sentence-transformers not installed. Semantic check using lexical fallback.")
            self.model = None

        if reranker_model is not None:
            self.reranker = reranker_model
        elif CrossEncoder is not None:
            try:
                print(f"  Loading Semantic Reranker ({reranker_model_name})...")
                self.reranker = CrossEncoder(reranker_model_name)
            except Exception as exc:
                print(f"  Warning: semantic reranker unavailable ({exc}). Using lexical reranking.")
                self.reranker = None
        else:
            self.reranker = None

    def _split_into_claims(self, text: str) -> list[str]:
        """Split generated text into meaningful factual claims."""
        claims = []
        for _, _, sentence in _split_sentences(text):
            cleaned = sentence.strip()
            cleaned = re.sub(r"^[-*]\s+", "", cleaned)
            cleaned = re.sub(r"^\d+\.\s+", "", cleaned)
            cleaned = re.sub(r"^#+\s*", "", cleaned)
            cleaned = cleaned.strip(" -")

            if not self._is_claim_candidate(cleaned):
                continue

            claims.append(cleaned)

        return claims

    def _is_claim_candidate(self, sentence: str) -> bool:
        """Filter out headers and low-signal boilerplate sentences."""
        if len(sentence) < 12:
            return False
        if sentence.endswith(":"):
            return False

        lower_sentence = sentence.lower()
        if lower_sentence in {
            "key findings",
            "medication review",
            "risk flags",
            "suggested follow-up",
            "what your results show",
            "your medications",
            "what to watch for",
            "questions to ask your doctor",
        }:
            return False

        boilerplate_patterns = (
            "talk to your doctor",
            "discuss this with your doctor",
            "follow up with your doctor",
            "ask your doctor",
        )
        if any(pattern in lower_sentence for pattern in boilerplate_patterns):
            anchor_tokens = self._extract_anchor_tokens(sentence)
            if len(anchor_tokens) <= 1 and not re.search(r"\d", sentence):
                return False

        return True

    def _collect_evidence_sentences(
        self,
        retrieved_chunks: list[dict],
        patient_context: dict | None,
    ) -> list[SemanticEvidence]:
        """Collect sentence-level evidence from chunks and patient context."""
        evidence_items = []

        for chunk in retrieved_chunks:
            source_id_parts = [
                str(chunk.get("note_id", "")),
                str(chunk.get("resource_type", "")),
                str(chunk.get("resource_id", "")),
            ]
            source_id = ":".join([part for part in source_id_parts if part]) or "chunk"
            source_text = str(chunk.get("note_text", ""))

            for _, _, sentence in _split_sentences(source_text):
                if not sentence.strip():
                    continue
                evidence_items.append(
                    SemanticEvidence(
                        sentence_text=sentence.strip(),
                        source_kind="retrieved_chunk",
                        source_id=source_id,
                        source_text=source_text,
                    )
                )

        rendered_context = _render_patient_context(patient_context)
        if rendered_context:
            source_id = str(patient_context.get("patient_id", "patient_context"))
            for _, _, sentence in _split_sentences(rendered_context):
                if not sentence.strip():
                    continue
                evidence_items.append(
                    SemanticEvidence(
                        sentence_text=sentence.strip(),
                        source_kind="patient_context",
                        source_id=source_id,
                        source_text=rendered_context,
                    )
                )

        return evidence_items

    def _extract_anchor_tokens(self, text: str) -> set[str]:
        """Extract semantic anchor tokens for claim/evidence comparison."""
        tokens = _tokenize_context(text)
        return {token for token in tokens if token not in SEMANTIC_GENERIC_TOKENS}

    def _extract_assertion_state(self, text: str) -> dict:
        """Extract coarse assertion cues from free text."""
        lowered = text.lower()
        negated = bool(
            re.search(
                r"\b(?:denies|denied|no|not|without|negative for|free of)\b",
                lowered,
            )
        )
        historical = bool(
            re.search(
                r"\b(?:history of|historical|previously|prior|past|resolved|former)\b",
                lowered,
            )
        )
        current = bool(
            re.search(
                r"\b(?:currently|current|active|presenting with|has|reports|showed|shows|is|are)\b",
                lowered,
            )
        )
        uncertain = bool(
            re.search(r"\b(?:possible|possibly|likely|suspected|concern for|may|might)\b", lowered)
        )

        return {
            "negated": negated,
            "historical": historical,
            "current": current,
            "uncertain": uncertain,
        }

    def _assertion_conflicts(self, claim: str, evidence: SemanticEvidence) -> bool:
        """Return True when claim and evidence have incompatible assertion cues."""
        claim_state = self._extract_assertion_state(claim)
        evidence_state = self._extract_assertion_state(evidence.sentence_text)

        if claim_state["negated"] != evidence_state["negated"]:
            if claim_state["negated"] or evidence_state["negated"]:
                return True

        if claim_state["historical"] and evidence_state["current"] and not evidence_state["historical"]:
            return True
        if evidence_state["historical"] and claim_state["current"] and not claim_state["historical"]:
            return True

        return False

    def _token_overlap_score(self, left: str, right: str) -> float:
        """Compute token overlap between two strings."""
        left_tokens = self._extract_anchor_tokens(left)
        right_tokens = self._extract_anchor_tokens(right)

        if not left_tokens or not right_tokens:
            return 0.0

        overlap = left_tokens & right_tokens
        return len(overlap) / max(len(left_tokens), 1)

    def _anchor_overlap_score(self, claim: str, evidence: SemanticEvidence) -> float:
        """Compute anchor overlap between claim and evidence sentence."""
        return self._token_overlap_score(claim, evidence.sentence_text)

    def _lexical_similarity(self, left: str, right: str) -> float:
        """Lexical fallback similarity used when transformer models are unavailable."""
        left_tokens = _tokenize_context(left)
        right_tokens = _tokenize_context(right)
        if not left_tokens or not right_tokens:
            return 0.0

        overlap = len(left_tokens & right_tokens)
        denominator = max(len(left_tokens | right_tokens), 1)
        return overlap / denominator

    def _normalize_model_score(self, score: float) -> float:
        """Normalize model scores into a 0-1 range."""
        numeric_score = float(score)
        if 0.0 <= numeric_score <= 1.0:
            return numeric_score
        return 1.0 / (1.0 + math.exp(-numeric_score))

    def _score_claims_bi_encoder(
        self,
        claims: list[str],
        evidence_items: list[SemanticEvidence],
    ) -> list[list[float]]:
        """Score all claims against all evidence sentences with the bi-encoder."""
        if not claims or not evidence_items:
            return []

        evidence_texts = [item.sentence_text for item in evidence_items]
        if self.model is not None and util is not None:
            claim_embeddings = self.model.encode(claims, convert_to_tensor=True)
            evidence_embeddings = self.model.encode(evidence_texts, convert_to_tensor=True)
            cosine_scores = util.cos_sim(claim_embeddings, evidence_embeddings)
            return [
                [float(cosine_scores[row_index][col_index]) for col_index in range(len(evidence_texts))]
                for row_index in range(len(claims))
            ]

        return [
            [self._lexical_similarity(claim, evidence_text) for evidence_text in evidence_texts]
            for claim in claims
        ]

    def _score_pairs_cross_encoder(self, claim: str, evidence_items: list[SemanticEvidence]) -> list[float]:
        """Rerank shortlisted claim/evidence pairs."""
        if not evidence_items:
            return []

        if self.reranker is not None:
            raw_scores = self.reranker.predict([(claim, item.sentence_text) for item in evidence_items])
            return [self._normalize_model_score(score) for score in raw_scores]

        scores = []
        for item in evidence_items:
            anchor_score = self._anchor_overlap_score(claim, item)
            lexical_score = self._lexical_similarity(claim, item.sentence_text)
            scores.append((0.65 * anchor_score) + (0.35 * lexical_score))
        return scores

    def _select_shortlist(
        self,
        claim: str,
        evidence_items: list[SemanticEvidence],
        bi_scores: list[float],
    ) -> list[tuple[SemanticEvidence, float]]:
        """Select the top-k evidence sentences for one claim."""
        ranked = sorted(
            zip(evidence_items, bi_scores),
            key=lambda item: item[1],
            reverse=True,
        )
        shortlist = ranked[: min(self.shortlist_k, len(ranked))]

        if not shortlist:
            return []

        if shortlist[0][1] <= 0.0:
            claim_tokens = self._extract_anchor_tokens(claim)
            anchor_ranked = sorted(
                (
                    (item, self._anchor_overlap_score(claim, item))
                    for item in evidence_items
                ),
                key=lambda pair: pair[1],
                reverse=True,
            )
            shortlist = [
                (item, score)
                for item, score in anchor_ranked[: min(self.shortlist_k, len(anchor_ranked))]
                if score > 0 or not claim_tokens
            ]

        return shortlist

    def _supports_claim(
        self,
        claim: str,
        evidence: SemanticEvidence | None,
        bi_score: float | None,
        cross_score: float | None,
    ) -> tuple[bool, str]:
        """Decide whether a claim is supported by its best evidence sentence."""
        if evidence is None:
            return False, "no_evidence_found"

        if self._assertion_conflicts(claim, evidence):
            return False, "assertion_mismatch"

        anchor_score = self._anchor_overlap_score(claim, evidence)
        claim_anchor_tokens = self._extract_anchor_tokens(claim)

        if claim_anchor_tokens and anchor_score <= 0.0:
            return False, "anchor_mismatch"

        if bi_score is None or cross_score is None:
            return False, "no_evidence_found"

        semantic_threshold = self.threshold if self.reranker is not None else self.reranker_threshold
        if bi_score < self.bi_encoder_threshold and cross_score < semantic_threshold:
            return False, "low_semantic_similarity"

        if cross_score < semantic_threshold:
            return False, "low_semantic_similarity"

        return True, "supported"

    def _should_suppress_for_numeric_findings(
        self,
        claim: str,
        evidence: SemanticEvidence | None,
        numeric_findings: list[dict] | None,
    ) -> bool:
        """Suppress duplicate semantic warnings for numeric-only mismatches."""
        if not numeric_findings or evidence is None:
            return False

        normalized_claim = claim.strip().lower()
        if not any(
            normalized_claim == str(finding.get("claim_sentence", "")).strip().lower()
            for finding in numeric_findings
        ):
            return False

        return self._anchor_overlap_score(claim, evidence) > 0.0

    def _build_finding(
        self,
        claim: str,
        reason: str,
        evidence: SemanticEvidence | None,
        bi_score: float | None,
        cross_score: float | None,
    ) -> dict:
        """Build a structured semantic finding."""
        return {
            "claim_text": claim,
            "claim_sentence": claim,
            "reason": reason,
            "best_candidate_source": evidence.sentence_text if evidence else "",
            "best_candidate_source_id": evidence.source_id if evidence else "",
            "bi_encoder_score": _safe_float(bi_score),
            "cross_encoder_score": _safe_float(cross_score),
        }

    def verify_detailed(
        self,
        generated_text: str,
        retrieved_chunks: list[dict],
        patient_context: dict | None = None,
        numeric_findings: list[dict] | None = None,
    ) -> list[dict]:
        """Check semantic support and return structured unsupported findings."""
        claims = self._split_into_claims(generated_text)
        if not claims:
            return []

        evidence_items = self._collect_evidence_sentences(retrieved_chunks, patient_context)
        if not evidence_items:
            return [
                self._build_finding(
                    claim=claim,
                    reason="no_evidence_found",
                    evidence=None,
                    bi_score=None,
                    cross_score=None,
                )
                for claim in claims
            ]

        bi_score_matrix = self._score_claims_bi_encoder(claims, evidence_items)
        findings = []

        for claim_index, claim in enumerate(claims):
            bi_scores = bi_score_matrix[claim_index] if bi_score_matrix else []
            shortlist = self._select_shortlist(claim, evidence_items, bi_scores)

            if not shortlist:
                reason = "anchor_mismatch" if self._extract_anchor_tokens(claim) else "no_evidence_found"
                finding = self._build_finding(
                    claim=claim,
                    reason=reason,
                    evidence=None,
                    bi_score=None,
                    cross_score=None,
                )
                if not self._should_suppress_for_numeric_findings(claim, None, numeric_findings):
                    findings.append(finding)
                continue

            shortlisted_items = [item for item, _ in shortlist]
            cross_scores = self._score_pairs_cross_encoder(claim, shortlisted_items)
            combined = []
            for (item, bi_score), cross_score in zip(shortlist, cross_scores):
                combined_score = (0.4 * bi_score) + (0.6 * cross_score)
                combined.append((item, bi_score, cross_score, combined_score))

            best_item, best_bi_score, best_cross_score, _ = max(
                combined,
                key=lambda item: item[3],
            )
            is_supported, reason = self._supports_claim(
                claim=claim,
                evidence=best_item,
                bi_score=best_bi_score,
                cross_score=best_cross_score,
            )

            if is_supported:
                continue

            if self._should_suppress_for_numeric_findings(claim, best_item, numeric_findings):
                continue

            findings.append(
                self._build_finding(
                    claim=claim,
                    reason=reason,
                    evidence=best_item,
                    bi_score=best_bi_score,
                    cross_score=best_cross_score,
                )
            )

        return findings

    def verify(
        self,
        generated_text: str,
        retrieved_chunks: list[dict],
        patient_context: dict | None = None,
        numeric_findings: list[dict] | None = None,
    ) -> list[str]:
        """Backward-compatible wrapper returning unsupported claim text."""
        findings = self.verify_detailed(
            generated_text=generated_text,
            retrieved_chunks=retrieved_chunks,
            patient_context=patient_context,
            numeric_findings=numeric_findings,
        )
        return [finding["claim_text"] for finding in findings]


def log_provenance(
    query: str,
    mode: str,
    chunks: list[dict],
    generated_text: str,
    unsupported_numbers: list[float | str],
    unsupported_claims: list[str],
    retrieval_ms: int,
    generation_ms: int,
    numeric_findings: list[dict] | None = None,
    semantic_findings: list[dict] | None = None,
) -> None:
    """Log the execution details to the audit CSV for traceability."""
    os.makedirs(DATA_DIR, exist_ok=True)
    log_path = os.path.join(DATA_DIR, AUDIT_LOG_FILE)

    file_exists = os.path.isfile(log_path)
    chunk_ids = [f"{c.get('resource_type', 'Note')}:{c.get('resource_id', 'Unknown')}" for c in chunks]

    row = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "mode": mode,
        "retrieved_chunk_ids": "|".join(chunk_ids),
        "generated_text": generated_text,
        "generated_text_length": len(generated_text),
        "unsupported_numbers": ",".join(map(str, unsupported_numbers)) if unsupported_numbers else "None",
        "unsupported_claims_count": len(unsupported_claims),
        "unsupported_claims": _json_dumps_compact(unsupported_claims),
        "numeric_findings_json": _json_dumps_compact(numeric_findings or []),
        "semantic_findings_json": _json_dumps_compact(semantic_findings or []),
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
