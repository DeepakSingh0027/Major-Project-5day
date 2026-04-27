# AGENTS.md

Guidelines for AI coding agents operating in this repository.

## Project Overview

Python data-extraction project for processing **Synthea-generated FHIR
(Fast Healthcare Interoperability Resources)** patient bundles. The scripts
read FHIR Bundle JSON files from `synthea/output/fhir/`, extract Patient,
Condition, Observation, and MedicationRequest resources, and write them to
CSV files in `datasets/`.

# Tasks

Converts the FHIR files into structured datasets.

## Task 5 — Parse FHIR Bundle JSON Files [DONE]

Goal: Convert JSON → tables.

Status: **Complete.** `extract_data.py` was refactored in-place.

### What was done

1. **Restructured** the flat procedural script into functions with a
   `main()` entry point and `if __name__ == "__main__":` guard.
2. **Enriched CSV schemas** — each output CSV now captures substantially
   more fields from the FHIR resources:

   | CSV               | New columns added                                                              |
   |-------------------|--------------------------------------------------------------------------------|
   | `patients.csv`    | `first_name`, `last_name`, `race`, `ethnicity`, `marital_status`, `city`, `state`, `country` |
   | `conditions.csv`  | `code` (SNOMED), `system`, `onset_date`, `abatement_date`, `clinical_status`   |
   | `observations.csv`| `observation` (display), `code` (LOINC), `system`, `date`, `category`          |
   | `medications.csv` | `code` (RxNorm), `system`, `start_date`, `end_date`, `status`                  |

3. **Resolved `urn:uuid:` medication references** — a `build_medication_lookup()`
   function scans the bundle for standalone `Medication` resources and maps
   their IDs to coding info, so `medicationReference` entries resolve to
   human-readable names instead of raw UUIDs.
4. **Added error handling** — missing data directory, malformed JSON files,
   missing keys in bundles, and output directory creation.
5. **Fixed bugs** — removed misplaced print statement (formerly line 9),
   eliminated trailing whitespace (formerly line 73).

### Functions in `extract_data.py`

| Function                    | Purpose                                          |
|-----------------------------|--------------------------------------------------|
| `parse_patient(resource)`   | Extract fields from a Patient resource            |
| `parse_condition(resource, patient_id)` | Extract fields from a Condition resource |
| `parse_observation(resource, patient_id)` | Extract fields from an Observation resource |
| `parse_medication(resource, patient_id, med_lookup)` | Extract fields from a MedicationRequest |
| `build_medication_lookup(entries)` | Build `{id: coding_info}` map from Medication resources |
| `process_bundle(filepath)`  | Parse one JSON file, dispatch to resource parsers |
| `main()`                    | Orchestrate: validate dirs, iterate files, save CSVs |

---

## Task 6 — Maintain Relational Integrity [DONE]
Goal: Ensure dataset relationships work.
Example schema:

patients
   |
   ├── conditions
   ├── medications
   └── lab_results

Ensure:

patient_id exists in patients table

Remove broken rows if necessary.

Status: **Complete.** `extract_data.py` was updated to rigorously enforce referential integrity.

### What was done

1. **Robust ID Extraction:** Modified `parse_condition`, `parse_observation`, and `parse_medication` to extract `patient_id` directly from the resource's `subject.reference` field (e.g., `"urn:uuid:<id>"`). This prevents orphan rows caused by positional `patient_id` assignment if child resources appear before the Patient in the bundle.
2. **Post-Extraction Validation:** Added a `validate_integrity()` function that runs just before writing the DataFrames to CSVs. It performs the following cleanup:
   - **Deduplicates patients:** Drops duplicate rows in the `patients` table (keeping the first occurrence).
   - **Removes invalid IDs:** Drops rows in child tables (`conditions`, `observations`, `medications`) where `patient_id` is null or empty.
   - **Removes orphan rows:** Drops child rows where the `patient_id` does not exist in the valid set of IDs from the `patients` table.
3. **Validation Reporting:** Prints a summary of any integrity issues found and cleaned up.



Task 7 — Data Cleaning & Normalization [DONE]

Goal: Standardize dataset.

Status: **Complete.** `extract_data.py` was updated with a new `clean_and_normalize` function that standardizes timestamps, calculates integer ages relative to the dataset max date, replaces medical code URIs with short names (SNOMED, LOINC, RxNorm), and isolates lab data. The output schemas are clean and uniform.

Tasks:
Convert:
birthDate → age
timestamp → datetime

Normalize medical codes:
RxNorm
LOINC
SNOMED

Output final dataset:
patients.csv
conditions.csv
labs.csv
medications.csv

Task 8 — Create Structured Database [DONE]

Goal: Create database or dataframe store.

Status: **Complete.** Created `build_database.py` which loads the four CSV datasets into a local SQLite database (`datasets/fhir_knowledge_base.db`). The script defines an explicit schema with primary and foreign keys and sets up indexes for fast queries.

Options:
SQLite,PostgreSQL,CSV dataset

Example schema:
patients
conditions
medications
labs

This becomes the knowledge base.

## Task 9 - Clinical NLP Pipeline [DONE]

Goal: Extract medical entities from clinical notes.

Status: **Complete.** Created `extract_clinical_entities.py` to scan FHIR
bundle note text with `medspaCy`, apply ConText assertions, and write a
structured entity-level CSV. `build_database.py` was also extended so the
NLP output can be loaded into SQLite when available.

### What was done

1. **Added a clinical NLP script** - `extract_clinical_entities.py` reads
   bundle JSON files from `synthea/output/fhir/`, collects free-text
   `note[].text` content and narrative `text.div` sections, and normalizes
   them into note records for processing.
2. **Integrated medspaCy + ConText** - each note is passed through
   `medspacy.load()`, and extracted entities are enriched with contextual
   flags such as `is_negated`, `is_uncertain`, `is_historical`, and
   `is_family`.
3. **Created note and entity datasets** - the pipeline writes
   `datasets/clinical_notes.csv` for note-level text records and
   `datasets/clinical_entities.csv` for entity-level extractions with
   patient/resource linkage plus assertion and offset metadata.
4. **Extended the knowledge base schema** - `build_database.py` now creates
   `clinical_notes` and `clinical_entities` tables and loads the CSVs if
   they exist, while keeping the original four-table flow intact when the
   NLP step has not been run.
5. **Added graceful fallbacks** - missing FHIR directories, malformed JSON,
   absent note text, and missing `medspacy` installations are all handled
   cleanly with user-facing messages.

### Output schema

`clinical_notes.csv` contains:

- `note_id`
- `patient_id`
- `bundle_file`
- `resource_type`
- `resource_id`
- `note_source`
- `note_text`

`clinical_entities.csv` contains:

- `entity_id`
- `patient_id`
- `bundle_file`
- `resource_type`
- `resource_id`
- `note_id`
- `note_source`
- `note_text`
- `entity_text`
- `entity_label`
- `assertion`
- `is_negated`
- `is_uncertain`
- `is_historical`
- `is_family`
- `start_char`
- `end_char`

## Task 10 - Vector Search Knowledge Base [DONE]

Goal: Build a semantic retrieval system over clinical notes.

Status: **Complete.** Created `vector_search.py` to generate embeddings from
clinical note text with `sentence-transformers`, store them in a FAISS index,
and run semantic search queries such as `"patients with chest pain but no
hypertension"`.

### What was done

1. **Prepared note-level search input** - Task 9 now persists
   `datasets/clinical_notes.csv`, which gives the vector pipeline a clean
   note corpus to embed.
2. **Added embedding generation** - `vector_search.py` loads
   `clinical_notes.csv`, encodes `note_text` with the
   `all-MiniLM-L6-v2` sentence-transformer by default, and saves the
   embedding matrix to `datasets/clinical_note_embeddings.npy`.
3. **Stored vectors in FAISS** - the script builds a
   `faiss.IndexFlatL2` index and saves it to
   `datasets/clinical_note_index.faiss` for fast semantic retrieval.
4. **Saved retrieval metadata** - note metadata is written to
   `datasets/note_vector_metadata.csv` and index settings are stored in
   `datasets/vector_index_config.json`.
5. **Built semantic search** - the same script supports
   `--query "patients with chest pain but no hypertension"` and returns the
   closest note matches with patient and resource metadata.
6. **Aligned relational and vector layers** - `build_database.py` now loads
   `clinical_notes.csv` into SQLite so note IDs stay consistent across the
   relational knowledge base and the FAISS vector store.

### Vector artifacts

Task 10 writes:

- `datasets/clinical_note_embeddings.npy`
- `datasets/clinical_note_index.faiss`
- `datasets/note_vector_metadata.csv`
- `datasets/vector_index_config.json`

## Task 11 — LLM Integration [DONE]

Goal: Set up Ollama with a local model for low-latency, privacy-preserving inference.

Status: **Complete.** Created `llm_client.py` — a thin wrapper around the
Ollama HTTP API that handles model connectivity, generation, and latency
tracking.

### What was done

1. **Created Ollama HTTP client** — `OllamaClient` class wraps the
   `http://localhost:11434/api/generate` endpoint with configurable model,
   timeout, and temperature settings.
2. **Added health checks** — `is_available()` verifies server connectivity,
   `list_models()` enumerates locally pulled models, `model_is_pulled()`
   confirms the configured model is ready.
3. **Latency tracking** — each `generate()` call measures wall-clock
   inference time in milliseconds and includes it in the response dict.
4. **Graceful degradation** — connection errors, timeouts, and missing
   models are reported cleanly without crashing the pipeline.

### Configuration

- Default model: `llama3` (Ollama)
- Default base URL: `http://localhost:11434`
- Target latency: < 2.8s inference overhead

---

## Task 12 — Hybrid Retrieval [DONE]

Goal: Connect the existing FAISS index to a RAG controller supporting both
lexical (BM25) and dense vector search.

Status: **Complete.** Created `hybrid_retriever.py` which fuses FAISS dense
retrieval with BM25 lexical retrieval using weighted score fusion.

### What was done

1. **Built BM25 retriever** — `BM25Retriever` class tokenizes note text and
   implements Okapi BM25 scoring via the `rank_bm25` library.
2. **Wrapped FAISS retriever** — `DenseRetriever` class loads the existing
   vector artifacts from `vector_search.py` (no duplication).
3. **Implemented score fusion** — `HybridRetriever` combines dense and
   lexical scores using min-max normalization and weighted linear
   combination (default 60% dense, 40% lexical).
4. **Verified retrieval quality** — tested with clinical queries, confirmed
   the retriever returns 5-8 relevant chunks with correct ranking.

### Fusion algorithm

1. Run FAISS dense search (top 3×k candidates)
2. Run BM25 lexical search (top 3×k candidates)
3. Normalize dense scores (invert L2 distance) to [0,1]
4. Normalize lexical scores to [0,1]
5. Fused score = dense_weight × dense_norm + lexical_weight × lexical_norm
6. Return top-k by fused score

---

## Task 13 — Dual Prompting [DONE]

Goal: Draft clinician and patient prompt templates for dual-mode generation.

Status: **Complete.** Created `prompt_templates.py` with structured system
prompts and prompt builders, and `rag_controller.py` as the orchestrator
that ties all components together.

### What was done

1. **Clinician system prompt** — instructs the LLM to produce structured
   output with Key Findings, Medication Review, Risk Flags, and Suggested
   Follow-Up sections using standard medical terminology.
2. **Patient system prompt** — instructs the LLM to produce jargon-free
   explanations at ~7th grade reading level with sections: What Your Results
   Show, Your Medications, What to Watch For, Questions to Ask Your Doctor.
3. **Prompt builders** — `build_clinician_prompt()` and
   `build_patient_prompt()` format retrieved chunks and patient context
   into complete prompt strings.
4. **RAG controller** — `RAGController` class orchestrates the full
   retrieve → prompt → generate pipeline with CLI support for
   `--mode clinician|patient|both`.

### Mock data

`generate_mock_data.py` creates 20 synthetic clinical notes, 10 patients,
and all vector artifacts for standalone testing without Synthea.

---

## Task 14 — Numeric Verification [DONE]

Goal: Verify generated numeric claims against retrieved source evidence.

Status: **Complete.** `verifier.py` now contains a structured numeric
verification layer, and `rag_controller.py` runs it for both clinician and
patient outputs.

### What was done

1. **Added structured numeric mention extraction** — `NumericVerifier`
   now captures raw numeric text, normalized value, unit, category, local
   context, source IDs, and character spans.
2. **Expanded numeric parsing coverage** — the verifier handles decimals,
   percentages, comma-formatted values, slash expressions such as blood
   pressure, medication doses, durations, and age-like expressions.
3. **Matched against actual prompt evidence** — numeric claims are checked
   against both retrieved chunk text and rendered patient context, not only a
   flat concatenation of chunk text.
4. **Added detailed verifier findings** — unsupported claims now produce
   structured `numeric_findings` with reason codes, best candidate source,
   and numeric difference data.
5. **Integrated numeric warnings into the RAG flow** — `rag_controller.py`
   preserves backward-compatible `unsupported_numbers` while surfacing
   richer numeric warnings in CLI output.
6. **Added test coverage and benchmark data** — `tests/test_numeric_verifier.py`
   validates extraction, matching, integration cases, and a labeled numeric
   benchmark fixture.

### Validation

- `python3 -m unittest tests/test_numeric_verifier.py -v`
- Current labeled numeric benchmark result: `25/25 = 100%`

---

## Task 15 — Semantic Check [PARTIAL]

Goal: Flag generated atomic claims that are not supported by retrieved
clinical evidence.

Status: **Partially complete.** A baseline semantic verifier exists in
`verifier.py`, and it has now been upgraded to sentence-level evidence with
structured findings, but it should still be treated as an evolving Day 2
verification layer rather than a final clinical fact-checking system.

### What is currently implemented

1. **Sentence-level claim extraction** — generated output is split into
   factual claims while skipping markdown headers, section labels, and
   low-signal boilerplate.
2. **Sentence-level evidence collection** — retrieved chunks and patient
   context are split into evidence sentences for more targeted matching.
3. **Hybrid semantic scoring** — `SemanticVerifier` now supports a two-stage
   flow: bi-encoder similarity for shortlist selection plus reranking over
   shortlisted claim/evidence pairs, with lexical fallbacks when models are
   unavailable.
4. **Structured semantic findings** — unsupported claims return
   `semantic_findings` with reason codes, best candidate source, and both
   bi-encoder and reranker scores.
5. **Assertion-aware rejection rules** — negation and historical/current
   mismatches are explicitly flagged as semantic conflicts.
6. **Numeric/semantic coordination** — semantic warnings are suppressed when
   a sentence is already explained by a numeric-only mismatch.

### Current limitation

- This verifier improves unsupported-claim detection, but it still relies on
  lightweight lexical cues and model similarity rather than full medical
  entailment or ontology-aware reasoning.

---

## Task 16 — Provenance Log [PARTIAL]

Goal: Log generated text, evidence references, and verifier outcomes into a
traceable audit trail.

Status: **Partially complete.** Audit logging already exists in
`verifier.py`, and it now stores generated text plus serialized verifier
details, but it remains a CSV-based trace rather than a full review system.

### What is currently implemented

1. **CSV audit log creation** — `log_provenance()` writes run-level audit
   rows to `datasets/audit_log.csv`.
2. **Traceable request metadata** — each row captures timestamp, query,
   response mode, retrieved chunk IDs, retrieval time, and generation time.
3. **Stored generated output** — the log now records the full generated text
   in addition to `generated_text_length`.
4. **Serialized verifier details** — numeric and semantic findings are
   persisted as compact JSON strings so the CSV stays flat but still
   preserves structured traceability.
5. **Backward-compatible summary fields** — numeric and semantic flags plus
   unsupported-claim counts remain available for quick inspection.

### Current limitation

- The audit log is now more traceable, but it is still a row-level CSV and
  not yet a complete reviewer-facing audit workflow with filtering,
  dashboards, or human annotation tooling.

---

## Final Workflow

Member 1
Synthetic Data Generation
        |
        v
Member 2
FHIR -> Structured Dataset
        |
        v
Member 3
Clinical NLP + Vector Search
        |
        v
Member 4 (Day 1)
RAG Controller (LLM + Hybrid Retrieval + Dual Prompting)
        |
        v
Member 4 (Day 2)
Verifier & Audit (Numeric Verification + Semantic Check + Provenance Log)

Because each stage can simulate or mock its inputs, members can still work in
parallel even though the final deliverable connects all three stages.

### Repository Structure

```
Major-Project-5day/
├── build_database.py    # Generates SQLite DB from CSV files
├── extract_clinical_entities.py  # Clinical NLP pipeline for FHIR note text
├── extract_data.py      # Main ETL script - reads FHIR bundles, writes CSVs
├── generate_mock_data.py # Generates mock clinical data for standalone testing
├── hybrid_retriever.py  # Hybrid BM25 + FAISS retrieval with score fusion
├── inspect_bundle.py    # Utility - prints resource types in a single bundle
├── llm_client.py        # Ollama LLM HTTP client wrapper
├── prompt_templates.py  # Dual-mode prompt templates (clinician / patient)
├── rag_controller.py    # RAG orchestrator - retrieval + prompting + LLM
├── vector_search.py     # Builds FAISS index and runs semantic note search
├── verifier.py          # Numeric / semantic verification and audit logging
├── datasets/            # Output CSV files, vector artifacts, and SQLite database
├── tests/               # Unittest coverage and benchmark fixtures
├── readme.md            # Project README (currently empty)
└── AGENTS.md            # This file
```

The `synthea/output/fhir/` data source directory is **not committed** to the
repo. It must exist locally (generated by running the Synthea patient
simulator) before running the scripts.

### Dependencies

- **Python 3** (3.10+ recommended)
- **pandas** — required for ETL and CSV/database loading
- **medspacy** — required for Task 9 clinical note entity extraction
- **sentence-transformers** — required for Task 10 note embeddings and Day 2
  verifier models
- **faiss-cpu** — required for Task 10 vector storage and semantic search
- **rank-bm25** — required for Task 12 lexical retrieval
- **requests** — required for Task 11 Ollama HTTP client
- **Ollama** — required for Task 11 local LLM inference (install from https://ollama.com)
- **sqlite3** — built-in Python module

Install Python packages:

```bash
pip install pandas medspacy sentence-transformers faiss-cpu rank-bm25 requests
```

Install and configure Ollama:

```bash
# Install Ollama from https://ollama.com/download
# Then pull the model:
ollama pull llama3
```

---

## Build / Lint / Test Commands

### Running Scripts

```bash
# Main data extraction (requires synthea/output/fhir/ to exist)
python extract_data.py

# Build SQLite database (requires extract_data.py to run first)
python build_database.py

# Extract clinical entities from FHIR notes (requires medspacy)
python extract_clinical_entities.py

# Build the vector index over notes
python vector_search.py

# Run semantic search over indexed notes
python vector_search.py --query "patients with chest pain but no hypertension" --top-k 5

# Inspect a single FHIR bundle
python inspect_bundle.py

# Generate mock data for standalone testing (no Synthea needed)
python generate_mock_data.py

# Run hybrid retrieval (BM25 + FAISS)
python hybrid_retriever.py --query "chest pain and diabetes" --top-k 8

# Test Ollama LLM connectivity
python llm_client.py

# Run the full RAG pipeline (retrieve + prompt + generate)
python rag_controller.py --query "patient with chest pain and diabetes" --mode both --top-k 8
python rag_controller.py --query "lupus treatment plan" --mode clinician
python rag_controller.py --query "what do my lab results mean" --mode patient
```

### Tests

The repository now includes **unittest-based coverage** for the numeric
verifier, semantic verifier, provenance log, and verification output shape.

Current test commands:

```bash
# Run current verifier and audit tests
python3 -m unittest tests/test_numeric_verifier.py -v
python3 -m unittest tests/test_semantic_verifier.py -v
python3 -m unittest discover -s tests -v
```

There is still **no CI pipeline** yet. If broader tests are added later,
`pytest` can still be adopted, but the existing repo currently uses
`unittest`.

### Linting and Formatting

**No linter or formatter is currently configured.** If adding tooling, prefer:

```bash
# Linting
ruff check .
ruff check extract_data.py          # single file

# Formatting
ruff format .
ruff format extract_data.py         # single file

# Type checking
mypy extract_data.py
```

---

## Code Style Guidelines

These are derived from the existing codebase patterns. Follow them for
consistency when modifying or extending the project.

### Python Version and Typing

- Target **Python 3.10+** (uses `dict | None` union syntax in type hints).
- `extract_data.py` uses **type annotations** on all function signatures
  (parameter types and return types). Maintain this when adding or modifying
  functions.

### Imports

- Use **module-level imports only** (no inline/local imports).
- Group imports in **PEP 8 order**, separated by blank lines:
  1. Standard library (`json`, `os`)
  2. Third-party (`pandas as pd`)
  3. Local/project modules
- Prefer `import module` over `from module import name` unless specific
  names are needed.
- Use standard aliases: `pandas` as `pd`, `numpy` as `np`.

### Naming Conventions

| Element          | Convention    | Examples                                    |
|------------------|---------------|---------------------------------------------|
| Files            | `snake_case`  | `extract_data.py`, `inspect_bundle.py`      |
| Variables        | `snake_case`  | `data_dir`, `patient_id`, `patients_df`     |
| Functions        | `snake_case`  | `parse_patient`, `process_bundle`, `main`   |
| Classes          | `PascalCase`  | (none yet — follow PEP 8)                   |
| Constants        | `UPPER_SNAKE` | `DATA_DIR`, `OUTPUT_DIR`                    |
| DataFrames       | `_df` suffix  | `patients_df`, `conditions_df`              |
| Lists of dicts   | Plural nouns  | `patients`, `conditions`, `observations`    |
| Dict keys (ours) | `snake_case`  | `"patient_id"`, `"condition"`, `"value"`    |

### Strings

- Use **double quotes** (`"`) consistently for all strings.
- Use f-strings for interpolation when adding new code.

### Indentation and Formatting

- **4 spaces** per indent level (PEP 8 standard).
- Keep lines under **100 characters**.
- Use blank lines generously between logical sections.
- Use section-divider comments to separate major blocks:
  ```python
  # -----------------
  # Section Name
  # -----------------
  ```

### Error Handling

- The current codebase uses **defensive access** (`dict.get()`, `"key" in dict`)
  rather than try/except. Maintain this pattern for FHIR resource field access.
- When adding new code, use **try/except** for I/O operations (file reads,
  JSON parsing) and let callers handle domain-level errors.
- Avoid bare `except:` — always catch specific exceptions.
- Use `encoding="utf-8"` when opening files.

### File Organization

When writing new scripts, structure them as:

```python
"""Module docstring describing purpose."""

import stdlib_module
import third_party_module

# Constants / configuration

def main():
    """Main entry point."""
    ...

if __name__ == "__main__":
    main()
```

`extract_data.py` already follows this pattern. `inspect_bundle.py` still
lacks an `if __name__ == "__main__":` guard; refactor it when modifying.

### Comments and Documentation

- Add a **module-level docstring** to new files.
- Add **function docstrings** (Google style or reStructuredText).
- Use `# section divider` comments to separate logical blocks (existing pattern).
- Inline comments should explain *why*, not *what*.

### Data Patterns

- Accumulate extracted records as **lists of dicts**, then convert to
  `pd.DataFrame` at the end.
- Save DataFrames with `to_csv(..., index=False)`.
- Output CSV files go in the `datasets/` directory.

---

## Known Issues

1. ~~**Misplaced print** — fixed in Task 5. Print now occurs after CSVs
   are saved.~~
2. **No `.gitignore`** — large CSV files in `datasets/` and the
   `synthea/` data directory are not excluded from version control.
3. **Hardcoded paths** — both scripts use string-literal paths with no
   CLI argument parsing or environment variable support.
4. ~~**No error handling for missing directories** — fixed in Task 5.
   `extract_data.py` now validates `DATA_DIR` exists and exits cleanly.~~
5. ~~**Trailing whitespace** — fixed in Task 5.~~
