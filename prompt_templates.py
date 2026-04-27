"""Dual-output prompt templates for the RAG clinical decision system.

Contains system prompts and prompt builders for two output modes:
1. Clinician-facing: technical, structured, with medical terminology.
2. Patient-facing: jargon-free, simplified, at ~7th grade reading level.
"""


# -----------------
# System Prompts
# -----------------

CLINICIAN_SYSTEM_PROMPT = """You are a clinical decision-support assistant for licensed healthcare professionals. Your role is to analyze retrieved clinical evidence and generate a structured clinical summary.

INSTRUCTIONS:
1. Analyze the retrieved clinical note chunks carefully.
2. Cross-reference findings across multiple notes when available.
3. Focus on clinically significant information.
4. Use standard medical terminology and accepted abbreviations.
5. Flag any potential safety concerns immediately.

OUTPUT FORMAT — use exactly these sections:

## Key Findings
- List abnormal lab values with reference ranges
- List active diagnoses and their current status
- Note any clinically significant vital sign abnormalities

## Medication Review
- List current medications with doses
- Flag potential drug-drug interactions
- Flag medications that may need adjustment based on lab values
- Note any contraindications given current conditions

## Risk Flags
- Identify urgent or critical findings requiring immediate attention
- Flag any values outside critical thresholds
- Note any concerning trends across multiple visits

## Suggested Follow-Up
- Recommend next diagnostic steps
- Suggest specialist referrals if indicated
- Propose monitoring schedule for abnormal values

IMPORTANT RULES:
- Only state what the evidence supports. Do not fabricate findings.
- If information is insufficient, explicitly state what is missing.
- Always note the source note for each finding.
- Use ICD/SNOMED codes where available.
"""

PATIENT_SYSTEM_PROMPT = """You are a friendly health educator helping a patient understand their medical information. Your goal is to explain clinical findings in simple, clear language that anyone can understand.

INSTRUCTIONS:
1. Use everyday language — avoid medical jargon.
2. When you must use a medical term, explain it in parentheses.
3. Keep sentences short and direct.
4. Focus on what matters most to the patient.
5. Be reassuring but honest.
6. Target a 7th-grade reading level.

OUTPUT FORMAT — use exactly these sections:

## What Your Results Show
- Explain each key finding in plain language
- Use comparisons to help understanding (e.g., "Your blood sugar is higher than the healthy range")
- Explain what each number means for their health

## Your Medications
- List medications with simple explanations of what each one does
- Mention any important things to watch for
- Explain why each medication was prescribed

## What to Watch For
- List warning signs that need medical attention
- Explain when to call the doctor vs. go to the ER
- Keep this practical and actionable

## Questions to Ask Your Doctor
- Suggest 3-5 specific questions the patient should ask
- Focus on questions that help the patient understand their care plan

IMPORTANT RULES:
- Never use medical abbreviations without explaining them.
- Do not provide a diagnosis — summarize what the notes say.
- If something is unclear from the notes, say so honestly.
- Always encourage the patient to discuss findings with their doctor.
"""


# -----------------
# Prompt Builders
# -----------------

def _format_chunks(retrieved_chunks: list[dict]) -> str:
    """Format retrieved note chunks into a numbered context block."""
    if not retrieved_chunks:
        return "No clinical notes were retrieved."

    parts = []
    for chunk in retrieved_chunks:
        note_id = chunk.get("note_id", "unknown")
        patient_id = chunk.get("patient_id", "unknown")
        note_text = chunk.get("note_text", "")
        fused_score = chunk.get("fused_score", 0.0)

        parts.append(
            f"[Note {note_id} | Patient {patient_id} | "
            f"Relevance {fused_score:.3f}]\n{note_text}"
        )

    return "\n\n---\n\n".join(parts)


def _format_patient_context(patient_context: dict) -> str:
    """Format patient demographic and condition context."""
    if not patient_context:
        return "No patient context provided."

    lines = []

    if patient_context.get("patient_id"):
        lines.append(f"Patient ID: {patient_context['patient_id']}")
    if patient_context.get("first_name"):
        name = f"{patient_context.get('first_name', '')} {patient_context.get('last_name', '')}".strip()
        lines.append(f"Name: {name}")
    if patient_context.get("age"):
        lines.append(f"Age: {patient_context['age']}")
    if patient_context.get("gender"):
        lines.append(f"Gender: {patient_context['gender']}")

    # Active conditions
    conditions = patient_context.get("conditions", [])
    if conditions:
        lines.append(f"Active conditions: {', '.join(conditions)}")

    # Current medications
    medications = patient_context.get("medications", [])
    if medications:
        lines.append(f"Current medications: {', '.join(medications)}")

    return "\n".join(lines) if lines else "No patient context provided."


def build_clinician_prompt(
    query: str,
    retrieved_chunks: list[dict],
    patient_context: dict | None = None,
) -> str:
    """Build a clinician-facing RAG prompt.

    Args:
        query: The clinical question or patient lookup query.
        retrieved_chunks: List of retrieved note chunk dicts from
                          hybrid_retriever.
        patient_context: Optional dict with patient demographics and
                         condition summary.

    Returns:
        Formatted prompt string ready for LLM generation.
    """
    context_block = _format_patient_context(patient_context or {})
    chunks_block = _format_chunks(retrieved_chunks)

    return f"""CLINICAL QUERY: {query}

PATIENT CONTEXT:
{context_block}

RETRIEVED CLINICAL NOTES:
{chunks_block}

Based on the retrieved clinical notes above, provide a structured clinical decision-support summary following your output format. Focus on abnormal findings, medication interactions, and recommended follow-up actions."""


def build_patient_prompt(
    query: str,
    retrieved_chunks: list[dict],
    patient_context: dict | None = None,
) -> str:
    """Build a patient-facing RAG prompt.

    Args:
        query: The patient's question or concern.
        retrieved_chunks: List of retrieved note chunk dicts from
                          hybrid_retriever.
        patient_context: Optional dict with patient demographics.

    Returns:
        Formatted prompt string ready for LLM generation.
    """
    context_block = _format_patient_context(patient_context or {})
    chunks_block = _format_chunks(retrieved_chunks)

    return f"""PATIENT QUESTION: {query}

PATIENT INFORMATION:
{context_block}

MEDICAL NOTES (for reference):
{chunks_block}

Using the medical notes above, explain the findings and care plan to the patient in simple, everyday language. Follow your output format and remember to avoid jargon."""
