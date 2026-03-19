"""Extract clinical entities from FHIR note text using medspaCy and ConText.

This script scans Synthea-generated FHIR Bundle JSON files for free-text notes
and narrative sections, runs medspaCy over that text, and writes the extracted
entities to a CSV file for downstream analysis or database loading.
"""

import html
import json
import os
import re
import sys

import pandas as pd

try:
    import medspacy
except ImportError:
    medspacy = None


# -----------------
# Configuration
# -----------------
DATA_DIR = "synthea/output/fhir"
OUTPUT_DIR = "datasets"
NOTES_OUTPUT_FILE = "clinical_notes.csv"
OUTPUT_FILE = "clinical_entities.csv"
NOTE_COLUMNS = [
    "note_id",
    "patient_id",
    "bundle_file",
    "resource_type",
    "resource_id",
    "note_source",
    "note_text",
]
ENTITY_COLUMNS = [
    "entity_id",
    "patient_id",
    "bundle_file",
    "resource_type",
    "resource_id",
    "note_id",
    "note_source",
    "note_text",
    "entity_text",
    "entity_label",
    "assertion",
    "is_negated",
    "is_uncertain",
    "is_historical",
    "is_family",
    "start_char",
    "end_char",
]


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim the string."""
    return re.sub(r"\s+", " ", text).strip()


def strip_html_markup(text: str) -> str:
    """Remove XHTML tags from a FHIR narrative field."""
    without_tags = re.sub(r"<[^>]+>", " ", text)
    return normalize_whitespace(html.unescape(without_tags))


def resolve_reference_id(reference: str) -> str:
    """Convert a FHIR reference to its ID component."""
    if not reference:
        return ""
    if reference.startswith("urn:uuid:"):
        return reference.replace("urn:uuid:", "")
    if "/" in reference:
        return reference.rsplit("/", maxsplit=1)[-1]
    return reference


def extract_patient_id(resource: dict) -> str:
    """Extract the patient ID from a FHIR resource."""
    if resource.get("resourceType") == "Patient":
        return resource.get("id", "")

    for field_name in ["subject", "patient"]:
        reference = resource.get(field_name, {}).get("reference", "")
        patient_id = resolve_reference_id(reference)
        if patient_id:
            return patient_id

    return ""


def build_note_record(
    resource: dict,
    bundle_file: str,
    note_source: str,
    note_index: int,
    note_text: str,
) -> dict:
    """Build a structured note record ready for NLP processing."""
    resource_id = resource.get("id", "")
    bundle_stem = os.path.splitext(bundle_file)[0]

    return {
        "note_id": f"{bundle_stem}_{resource_id or 'unknown'}_{note_source}_{note_index}",
        "patient_id": extract_patient_id(resource),
        "bundle_file": bundle_file,
        "resource_type": resource.get("resourceType", ""),
        "resource_id": resource_id,
        "note_source": note_source,
        "note_text": note_text,
    }


def collect_note_records(resource: dict, bundle_file: str) -> list[dict]:
    """Collect all free-text notes and narrative text from a resource."""
    note_records = []

    for index, note in enumerate(resource.get("note", []), start=1):
        cleaned_text = normalize_whitespace(note.get("text", ""))
        if cleaned_text:
            note_records.append(
                build_note_record(resource, bundle_file, "annotation", index, cleaned_text)
            )

    narrative = resource.get("text", {}).get("div", "")
    cleaned_narrative = strip_html_markup(narrative) if narrative else ""
    if cleaned_narrative:
        note_records.append(
            build_note_record(resource, bundle_file, "narrative", 1, cleaned_narrative)
        )

    return note_records


def process_bundle(filepath: str) -> list[dict]:
    """Collect note records from a single FHIR Bundle JSON file."""
    with open(filepath, encoding="utf-8", errors="ignore") as file_obj:
        bundle = json.load(file_obj)

    bundle_file = os.path.basename(filepath)
    note_records = []

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        note_records.extend(collect_note_records(resource, bundle_file))

    return note_records


def collect_all_notes(data_dir: str) -> list[dict]:
    """Collect note records from all bundle files in a directory."""
    files = [filename for filename in os.listdir(data_dir) if filename.endswith(".json")]
    all_notes = []

    for filename in files:
        filepath = os.path.join(data_dir, filename)
        try:
            all_notes.extend(process_bundle(filepath))
        except json.JSONDecodeError as exc:
            print(f"Warning: skipping malformed JSON file '{filename}': {exc}")
        except KeyError as exc:
            print(f"Warning: skipping file '{filename}' due to missing key: {exc}")

    return all_notes


def get_context_flag(entity, attribute_name: str) -> bool:
    """Safely read a medspaCy ConText boolean flag from an entity."""
    try:
        return bool(getattr(entity._, attribute_name))
    except AttributeError:
        return False


def determine_assertion(
    is_negated: bool,
    is_uncertain: bool,
    is_historical: bool,
    is_family: bool,
) -> str:
    """Summarize ConText modifiers as a single assertion label."""
    if is_negated:
        return "negated"
    if is_uncertain:
        return "possible"
    if is_historical:
        return "historical"
    if is_family:
        return "family_history"
    return "present"


def build_nlp():
    """Load the default medspaCy pipeline."""
    if medspacy is None:
        print("Error: medspaCy is not installed.")
        print("Install it with: pip install medspacy")
        sys.exit(1)

    return medspacy.load()


def extract_entities_from_note(note_record: dict, nlp) -> list[dict]:
    """Run medspaCy over a note and return entity-level records."""
    doc = nlp(note_record["note_text"])
    entity_records = []

    for index, entity in enumerate(doc.ents, start=1):
        is_negated = get_context_flag(entity, "is_negated")
        is_uncertain = get_context_flag(entity, "is_uncertain")
        is_historical = get_context_flag(entity, "is_historical")
        is_family = get_context_flag(entity, "is_family")

        entity_records.append(
            {
                "entity_id": f"{note_record['note_id']}_{index}_{entity.start_char}_{entity.end_char}",
                "patient_id": note_record["patient_id"],
                "bundle_file": note_record["bundle_file"],
                "resource_type": note_record["resource_type"],
                "resource_id": note_record["resource_id"],
                "note_id": note_record["note_id"],
                "note_source": note_record["note_source"],
                "note_text": note_record["note_text"],
                "entity_text": entity.text,
                "entity_label": entity.label_,
                "assertion": determine_assertion(
                    is_negated,
                    is_uncertain,
                    is_historical,
                    is_family,
                ),
                "is_negated": is_negated,
                "is_uncertain": is_uncertain,
                "is_historical": is_historical,
                "is_family": is_family,
                "start_char": entity.start_char,
                "end_char": entity.end_char,
            }
        )

    return entity_records


def main():
    """Main entry point: collect notes, run NLP, and save the entity table."""
    if not os.path.isdir(DATA_DIR):
        print(f"Error: data directory '{DATA_DIR}' not found.")
        print("Run the Synthea patient simulator first to generate FHIR bundles.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = [filename for filename in os.listdir(DATA_DIR) if filename.endswith(".json")]
    if not files:
        print(f"No .json files found in '{DATA_DIR}'.")
        sys.exit(1)

    print(f"\nScanning {len(files)} FHIR bundle(s) for note text...\n")

    all_notes = collect_all_notes(DATA_DIR)

    print(f"Collected {len(all_notes)} note(s) for NLP processing.")

    notes_df = pd.DataFrame(all_notes, columns=NOTE_COLUMNS)
    notes_output_path = os.path.join(OUTPUT_DIR, NOTES_OUTPUT_FILE)
    notes_df.to_csv(notes_output_path, index=False)

    if not all_notes:
        entities_df = pd.DataFrame(columns=ENTITY_COLUMNS)
    else:
        nlp = build_nlp()
        all_entities = []
        for note_record in all_notes:
            all_entities.extend(extract_entities_from_note(note_record, nlp))

        entities_df = pd.DataFrame(all_entities, columns=ENTITY_COLUMNS)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    entities_df.to_csv(output_path, index=False)

    print(f"Extracted {len(entities_df)} clinical entity row(s).")
    print(f"Clinical note dataset saved successfully to '{notes_output_path}'")
    print(f"Clinical NLP dataset saved successfully to '{output_path}'")


if __name__ == "__main__":
    main()
