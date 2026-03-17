"""Extract structured data from Synthea-generated FHIR Bundle JSON files.

Reads all .json files from the FHIR data directory, extracts Patient,
Condition, Observation, and MedicationRequest resources, and writes
enriched CSV files to the datasets/ directory.
"""

import json
import os
import sys

import pandas as pd


# -----------------
# Configuration
# -----------------
DATA_DIR = "synthea/output/fhir"
OUTPUT_DIR = "datasets"


def parse_patient(resource: dict) -> dict:
    """Extract fields from a FHIR Patient resource.

    Args:
        resource: A FHIR Patient resource dict.

    Returns:
        A flat dict of patient fields for tabular output.
    """
    # Name — Synthea bundles store official name in name[0]
    first_name = ""
    last_name = ""
    names = resource.get("name", [])
    if names:
        first_given = names[0].get("given", [])
        first_name = first_given[0] if first_given else ""
        last_name = names[0].get("family", "")

    # Race and ethnicity from US Core extensions
    race = ""
    ethnicity = ""
    for ext in resource.get("extension", []):
        url = ext.get("url", "")
        if "us-core-race" in url:
            for sub in ext.get("extension", []):
                if sub.get("url") == "text":
                    race = sub.get("valueString", "")
        elif "us-core-ethnicity" in url:
            for sub in ext.get("extension", []):
                if sub.get("url") == "text":
                    ethnicity = sub.get("valueString", "")

    # Address — take the first address entry
    city = ""
    state = ""
    country = ""
    addresses = resource.get("address", [])
    if addresses:
        city = addresses[0].get("city", "")
        state = addresses[0].get("state", "")
        country = addresses[0].get("country", "")

    marital_status = ""
    ms = resource.get("maritalStatus")
    if ms:
        marital_status = ms.get("text", "")

    return {
        "patient_id": resource.get("id", ""),
        "first_name": first_name,
        "last_name": last_name,
        "gender": resource.get("gender", ""),
        "birthDate": resource.get("birthDate", ""),
        "race": race,
        "ethnicity": ethnicity,
        "marital_status": marital_status,
        "city": city,
        "state": state,
        "country": country,
    }


def _resolve_patient_id(resource: dict, fallback_id: str | None) -> str:
    """Extract patient_id from a resource's subject.reference field.

    FHIR child resources (Condition, Observation, MedicationRequest) carry
    a ``subject.reference`` that points back to the Patient, typically as
    ``urn:uuid:<id>``.  Using this is more robust than relying on resource
    ordering within the bundle.

    Args:
        resource: A FHIR resource dict that may contain a ``subject`` field.
        fallback_id: Value to return when subject.reference is absent.

    Returns:
        The resolved patient ID string.
    """
    ref = resource.get("subject", {}).get("reference", "")
    if ref:
        return ref.replace("urn:uuid:", "")
    return fallback_id if fallback_id is not None else ""


def parse_condition(resource: dict, patient_id: str | None) -> dict | None:
    """Extract fields from a FHIR Condition resource.

    Args:
        resource: A FHIR Condition resource dict.
        patient_id: Fallback patient ID if subject.reference is absent.

    Returns:
        A flat dict of condition fields, or None if code is missing.
    """
    if "code" not in resource:
        return None

    coding = resource["code"].get("coding", [{}])[0]

    # Clinical status (active, resolved, etc.)
    clinical_status = ""
    cs = resource.get("clinicalStatus")
    if cs:
        cs_coding = cs.get("coding", [{}])
        if cs_coding:
            clinical_status = cs_coding[0].get("code", "")

    return {
        "patient_id": _resolve_patient_id(resource, patient_id),
        "condition": coding.get("display", ""),
        "code": coding.get("code", ""),
        "system": coding.get("system", ""),
        "onset_date": resource.get("onsetDateTime", ""),
        "abatement_date": resource.get("abatementDateTime", ""),
        "clinical_status": clinical_status,
    }


def parse_observation(resource: dict, patient_id: str | None) -> dict | None:
    """Extract fields from a FHIR Observation resource.

    Args:
        resource: A FHIR Observation resource dict.
        patient_id: Fallback patient ID if subject.reference is absent.

    Returns:
        A flat dict of observation fields, or None if valueQuantity is missing.
    """
    if "valueQuantity" not in resource:
        return None

    vq = resource["valueQuantity"]
    coding = resource.get("code", {}).get("coding", [{}])[0]

    # Category (vital-signs, laboratory, etc.)
    category = ""
    cats = resource.get("category", [])
    if cats:
        cat_coding = cats[0].get("coding", [{}])
        if cat_coding:
            category = cat_coding[0].get("code", "")

    return {
        "patient_id": _resolve_patient_id(resource, patient_id),
        "observation": coding.get("display", ""),
        "code": coding.get("code", ""),
        "system": coding.get("system", ""),
        "value": vq.get("value", ""),
        "unit": vq.get("unit", ""),
        "date": resource.get("effectiveDateTime", ""),
        "category": category,
    }


def parse_medication(
    resource: dict, patient_id: str | None, med_lookup: dict
) -> dict | None:
    """Extract fields from a FHIR MedicationRequest resource.

    Resolves medicationReference URIs using a lookup dict built from
    Medication resources in the same bundle.

    Args:
        resource: A FHIR MedicationRequest resource dict.
        patient_id: Fallback patient ID if subject.reference is absent.
        med_lookup: Maps resource IDs to medication display names.

    Returns:
        A flat dict of medication fields, or None if no medication found.
    """
    med_name = ""
    med_code = ""
    med_system = ""

    if "medicationCodeableConcept" in resource:
        coding = resource["medicationCodeableConcept"].get("coding", [{}])[0]
        med_name = coding.get("display", "")
        med_code = coding.get("code", "")
        med_system = coding.get("system", "")
    elif "medicationReference" in resource:
        ref = resource["medicationReference"].get("reference", "")
        # Try resolving the urn:uuid: reference via the lookup dict
        ref_id = ref.replace("urn:uuid:", "")
        resolved = med_lookup.get(ref_id)
        if resolved:
            med_name = resolved.get("display", "")
            med_code = resolved.get("code", "")
            med_system = resolved.get("system", "")
        else:
            # Fallback: keep the raw reference so no data is silently lost
            med_name = ref

    if not med_name:
        return None

    # Dosage period (start/end dates from dosageInstruction or authoredOn)
    start_date = resource.get("authoredOn", "")
    end_date = ""
    dosage_list = resource.get("dosageInstruction", [])
    if dosage_list:
        timing = dosage_list[0].get("timing", {})
        repeat = timing.get("repeat", {})
        bounds = repeat.get("boundsPeriod", {})
        if bounds:
            start_date = bounds.get("start", start_date)
            end_date = bounds.get("end", "")

    return {
        "patient_id": _resolve_patient_id(resource, patient_id),
        "medication": med_name,
        "code": med_code,
        "system": med_system,
        "start_date": start_date,
        "end_date": end_date,
        "status": resource.get("status", ""),
    }


def build_medication_lookup(entries: list) -> dict:
    """Build a lookup dict from Medication resources in a bundle.

    Synthea bundles sometimes include standalone Medication resources
    that MedicationRequest entries reference via urn:uuid:. This builds
    a map of {resource_id: {display, code, system}} so we can resolve
    those references to human-readable names.

    Args:
        entries: The list of entry dicts from a FHIR Bundle.

    Returns:
        Dict mapping resource IDs to their coding info.
    """
    lookup = {}
    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Medication":
            continue
        res_id = resource.get("id", "")
        # Also map the fullUrl (which is what urn:uuid: references point to)
        full_url = entry.get("fullUrl", "").replace("urn:uuid:", "")
        coding = resource.get("code", {}).get("coding", [{}])[0]
        info = {
            "display": coding.get("display", ""),
            "code": coding.get("code", ""),
            "system": coding.get("system", ""),
        }
        if res_id:
            lookup[res_id] = info
        if full_url:
            lookup[full_url] = info
    return lookup


def validate_integrity(
    patients_df: pd.DataFrame,
    conditions_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    medications_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate referential integrity across the datasets.

    Ensures that:
    1. Patients are unique by patient_id.
    2. Child tables (conditions, observations, medications) have valid,
       non-null patient_ids.
    3. All patient_ids in child tables exist in the patients table (no orphans).

    Args:
        patients_df: The extracted patients.
        conditions_df: The extracted conditions.
        observations_df: The extracted observations.
        medications_df: The extracted medications.

    Returns:
        A tuple of cleaned DataFrames: (patients, conditions, observations, medications).
    """
    print("\nValidating relational integrity...")

    # 1. Deduplicate patients
    orig_patients_count = len(patients_df)
    patients_df = patients_df.drop_duplicates(subset=["patient_id"], keep="first")
    dropped_patients = orig_patients_count - len(patients_df)

    valid_patient_ids = set(patients_df["patient_id"].dropna())

    def clean_child_table(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
        if df.empty:
            return df, 0, 0

        orig_count = len(df)

        # Identify rows with empty/null patient_id
        is_empty = df["patient_id"].astype(str).str.strip().isin(["", "nan", "None"])
        df_valid_id = df[~is_empty]
        dropped_nulls = orig_count - len(df_valid_id)

        # Identify orphans
        is_orphan = ~df_valid_id["patient_id"].isin(valid_patient_ids)
        df_clean = df_valid_id[~is_orphan]
        dropped_orphans = len(df_valid_id) - len(df_clean)

        return df_clean, dropped_nulls, dropped_orphans

    conditions_df, cond_nulls, cond_orphans = clean_child_table(conditions_df)
    observations_df, obs_nulls, obs_orphans = clean_child_table(observations_df)
    medications_df, med_nulls, med_orphans = clean_child_table(medications_df)

    total_issues = (
        dropped_patients
        + cond_nulls + cond_orphans
        + obs_nulls + obs_orphans
        + med_nulls + med_orphans
    )

    if total_issues == 0:
        print("  No integrity issues found.")
    else:
        if dropped_patients > 0:
            print(f"  Dropped {dropped_patients} duplicate patient(s).")
        if cond_nulls > 0 or cond_orphans > 0:
            print(f"  Conditions: dropped {cond_nulls} null/empty IDs, {cond_orphans} orphan(s).")
        if obs_nulls > 0 or obs_orphans > 0:
            print(f"  Observations: dropped {obs_nulls} null/empty IDs, {obs_orphans} orphan(s).")
        if med_nulls > 0 or med_orphans > 0:
            print(f"  Medications: dropped {med_nulls} null/empty IDs, {med_orphans} orphan(s).")

    return patients_df, conditions_df, observations_df, medications_df


def clean_and_normalize(
    patients_df: pd.DataFrame,
    conditions_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    medications_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean and normalize the datasets.

    Tasks:
    1. Standardize dates to datetime format.
    2. Calculate patient age relative to the max dataset date.
    3. Normalize medical coding systems (SNOMED, LOINC, RxNorm).
    4. Filter observations to include only 'laboratory' category -> labs_df.

    Args:
        patients_df: The validated patients DataFrame.
        conditions_df: The validated conditions DataFrame.
        observations_df: The validated observations DataFrame.
        medications_df: The validated medications DataFrame.

    Returns:
        A tuple of (patients_df, conditions_df, labs_df, medications_df).
    """
    print("\nCleaning and normalizing data...")

    # 1. Standardize Timestamps
    if not conditions_df.empty:
        conditions_df["onset_date"] = pd.to_datetime(conditions_df["onset_date"], errors="coerce", utc=True)
        conditions_df["abatement_date"] = pd.to_datetime(conditions_df["abatement_date"], errors="coerce", utc=True)
    if not observations_df.empty:
        observations_df["date"] = pd.to_datetime(observations_df["date"], errors="coerce", utc=True)
    if not medications_df.empty:
        medications_df["start_date"] = pd.to_datetime(medications_df["start_date"], errors="coerce", utc=True)
        medications_df["end_date"] = pd.to_datetime(medications_df["end_date"], errors="coerce", utc=True)

    # 2. Calculate Age
    # Find global maximum date across the temporal tables
    max_dates = []
    if not conditions_df.empty and not conditions_df["onset_date"].isna().all():
        max_dates.append(conditions_df["onset_date"].max())
    if not observations_df.empty and not observations_df["date"].isna().all():
        max_dates.append(observations_df["date"].max())
    if not medications_df.empty and not medications_df["start_date"].isna().all():
        max_dates.append(medications_df["start_date"].max())

    if max_dates:
        global_max_date = max(max_dates)
    else:
        global_max_date = pd.Timestamp.utcnow()

    if not patients_df.empty and "birthDate" in patients_df.columns:
        birth_dates = pd.to_datetime(patients_df["birthDate"], errors="coerce", utc=True)
        patients_df["age"] = (global_max_date - birth_dates).dt.days // 365.25
        patients_df["age"] = patients_df["age"].fillna(0).astype(int)
        patients_df = patients_df.drop(columns=["birthDate"])

    # 3. Normalize medical codes
    def normalize_system(sys_series: pd.Series) -> pd.Series:
        return sys_series.apply(
            lambda x: "SNOMED" if "snomed" in str(x).lower() else
                      "LOINC" if "loinc" in str(x).lower() else
                      "RxNorm" if "rxnorm" in str(x).lower() else x
        )

    if not conditions_df.empty and "system" in conditions_df.columns:
        conditions_df["system"] = normalize_system(conditions_df["system"])
    if not observations_df.empty and "system" in observations_df.columns:
        observations_df["system"] = normalize_system(observations_df["system"])
    if not medications_df.empty and "system" in medications_df.columns:
        medications_df["system"] = normalize_system(medications_df["system"])

    # 4. Filter Observations to Labs
    if not observations_df.empty and "category" in observations_df.columns:
        labs_df = observations_df[observations_df["category"] == "laboratory"].copy()
    else:
        labs_df = observations_df.copy()

    print(f"  Labs (filtered from observations): {len(labs_df)}")

    return patients_df, conditions_df, labs_df, medications_df


def process_bundle(filepath: str) -> tuple:
    """Parse a single FHIR Bundle JSON file and extract all resources.

    Args:
        filepath: Path to the .json bundle file.

    Returns:
        Tuple of (patients, conditions, observations, medications) lists.
    """
    patients = []
    conditions = []
    observations = []
    medications = []

    with open(filepath, encoding="utf-8", errors="ignore") as f:
        bundle = json.load(f)

    entries = bundle.get("entry", [])

    # Build medication reference lookup before iterating
    med_lookup = build_medication_lookup(entries)

    patient_id = None

    for entry in entries:
        resource = entry.get("resource", {})
        rtype = resource.get("resourceType", "")

        # -----------------
        # Patient
        # -----------------
        if rtype == "Patient":
            patient_id = resource.get("id", "")
            patients.append(parse_patient(resource))

        # -----------------
        # Condition
        # -----------------
        elif rtype == "Condition":
            record = parse_condition(resource, patient_id)
            if record:
                conditions.append(record)

        # -----------------
        # Observation
        # -----------------
        elif rtype == "Observation":
            record = parse_observation(resource, patient_id)
            if record:
                observations.append(record)

        # -----------------
        # MedicationRequest
        # -----------------
        elif rtype == "MedicationRequest":
            record = parse_medication(resource, patient_id, med_lookup)
            if record:
                medications.append(record)

    return patients, conditions, observations, medications


def main():
    """Main entry point: read FHIR bundles, extract resources, save CSVs."""
    # -----------------
    # Validate input directory
    # -----------------
    if not os.path.isdir(DATA_DIR):
        print(f"Error: data directory '{DATA_DIR}' not found.")
        print("Run the Synthea patient simulator first to generate FHIR bundles.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------
    # Collect records from all bundles
    # -----------------
    all_patients = []
    all_conditions = []
    all_observations = []
    all_medications = []

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]

    if not files:
        print(f"No .json files found in '{DATA_DIR}'.")
        sys.exit(1)

    print(f"\nProcessing {len(files)} FHIR bundle(s) from '{DATA_DIR}'...\n")

    for filename in files:
        filepath = os.path.join(DATA_DIR, filename)
        try:
            patients, conditions, observations, medications = process_bundle(filepath)
            all_patients.extend(patients)
            all_conditions.extend(conditions)
            all_observations.extend(observations)
            all_medications.extend(medications)
        except json.JSONDecodeError as e:
            print(f"Warning: skipping malformed JSON file '{filename}': {e}")
            continue
        except KeyError as e:
            print(f"Warning: skipping file '{filename}' due to missing key: {e}")
            continue

    # -----------------
    # Summary
    # -----------------
    print("Extraction complete\n")
    print(f"  Patients:     {len(all_patients)}")
    print(f"  Conditions:   {len(all_conditions)}")
    print(f"  Observations: {len(all_observations)}")
    print(f"  Medications:  {len(all_medications)}")

    # -----------------
    # Convert to DataFrames and save
    # -----------------
    patients_df = pd.DataFrame(all_patients)
    conditions_df = pd.DataFrame(all_conditions)
    observations_df = pd.DataFrame(all_observations)
    medications_df = pd.DataFrame(all_medications)

    patients_df, conditions_df, observations_df, medications_df = validate_integrity(
        patients_df, conditions_df, observations_df, medications_df
    )

    patients_df, conditions_df, labs_df, medications_df = clean_and_normalize(
        patients_df, conditions_df, observations_df, medications_df
    )

    patients_df.to_csv(os.path.join(OUTPUT_DIR, "patients.csv"), index=False)
    conditions_df.to_csv(os.path.join(OUTPUT_DIR, "conditions.csv"), index=False)
    labs_df.to_csv(os.path.join(OUTPUT_DIR, "labs.csv"), index=False)
    medications_df.to_csv(os.path.join(OUTPUT_DIR, "medications.csv"), index=False)

    print(f"\nDatasets saved successfully to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
