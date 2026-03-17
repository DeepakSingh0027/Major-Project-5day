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

### Repository Structure

```
Major-Project-5day/
├── build_database.py    # Generates SQLite DB from CSV files
├── extract_data.py      # Main ETL script — reads FHIR bundles, writes CSVs
├── inspect_bundle.py    # Utility — prints resource types in a single bundle
├── datasets/            # Output CSV files and SQLite database
├── readme.md            # Project README (currently empty)
└── AGENTS.md            # This file
```

The `synthea/output/fhir/` data source directory is **not committed** to the
repo. It must exist locally (generated by running the Synthea patient
simulator) before running the scripts.

### Dependencies

- **Python 3** (3.8+ recommended)
- **pandas** — the only third-party dependency
- **sqlite3** — built-in Python module

There is no `requirements.txt` or `pyproject.toml` yet. Install manually:

```bash
pip install pandas
```

---

## Build / Lint / Test Commands

### Running Scripts

```bash
# Main data extraction (requires synthea/output/fhir/ to exist)
python extract_data.py

# Build SQLite database (requires extract_data.py to run first)
python build_database.py

# Inspect a single FHIR bundle
python inspect_bundle.py
```

### Tests

**No test suite exists yet.** There are no test files, no pytest/unittest
configuration, and no CI pipeline.

If tests are added, follow these conventions:

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_extract.py

# Run a single test function
pytest tests/test_extract.py::test_patient_extraction

# Run tests matching a keyword
pytest -k "patient"

# Run with verbose output
pytest -v
```

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
