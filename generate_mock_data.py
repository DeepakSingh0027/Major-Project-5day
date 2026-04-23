"""Generate mock clinical datasets for standalone RAG controller testing.

Creates synthetic clinical notes, patient records, and vector search
artifacts so the RAG pipeline can be exercised without running the full
Synthea → extract → NLP → vector pipeline.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

# -----------------
# Configuration
# -----------------
OUTPUT_DIR = "datasets"
MOCK_NOTE_COUNT = 20


# -----------------
# Mock Clinical Notes
# -----------------
MOCK_NOTES = [
    {
        "patient_id": "p001",
        "note_text": (
            "Patient is a 67-year-old male presenting with chest pain radiating "
            "to the left arm. History of hypertension and type 2 diabetes mellitus. "
            "Current medications include metformin 1000mg twice daily and lisinopril "
            "20mg daily. Blood pressure 158/92 mmHg. Heart rate 88 bpm. ECG shows "
            "ST-segment depression in leads V4-V6. Troponin I elevated at 0.08 ng/mL. "
            "Recommend cardiology consult and serial troponins."
        ),
    },
    {
        "patient_id": "p001",
        "note_text": (
            "Lab results: HbA1c 8.2% (above target of 7.0%). Fasting glucose 186 mg/dL. "
            "LDL cholesterol 142 mg/dL. HDL cholesterol 38 mg/dL. Triglycerides 220 mg/dL. "
            "Creatinine 1.3 mg/dL, eGFR 58 mL/min (stage 3a CKD). Consider adding "
            "atorvastatin for lipid management. Monitor renal function closely given "
            "metformin use with declining eGFR."
        ),
    },
    {
        "patient_id": "p002",
        "note_text": (
            "Patient is a 45-year-old female with a 3-day history of productive cough, "
            "fever (101.2°F), and shortness of breath. Denies chest pain. Non-smoker. "
            "Lung auscultation reveals crackles in the right lower lobe. SpO2 94% on "
            "room air. Chest X-ray shows right lower lobe consolidation consistent with "
            "community-acquired pneumonia. Started on azithromycin 500mg day 1, then "
            "250mg daily for 4 days."
        ),
    },
    {
        "patient_id": "p002",
        "note_text": (
            "Follow-up visit: Patient reports improvement in cough and fever resolved. "
            "SpO2 improved to 97%. Completing antibiotic course. No adverse drug reactions "
            "reported. Continue current management. Return if symptoms worsen or new "
            "symptoms develop."
        ),
    },
    {
        "patient_id": "p003",
        "note_text": (
            "Patient is a 72-year-old female with progressive memory loss over the past "
            "18 months. Family reports difficulty with daily tasks, getting lost in "
            "familiar places, and word-finding difficulty. MMSE score 22/30. MRI shows "
            "bilateral hippocampal atrophy. No evidence of vascular disease. Assessment: "
            "probable Alzheimer disease, mild stage. Started donepezil 5mg daily. "
            "Referral to neurology for further evaluation."
        ),
    },
    {
        "patient_id": "p003",
        "note_text": (
            "Current medications: donepezil 5mg daily, amlodipine 5mg daily, "
            "omeprazole 20mg daily. Patient tolerating donepezil well. No GI side "
            "effects. Blood pressure controlled at 128/78 mmHg. Caregiver reports "
            "slight improvement in alertness. Continue current regimen."
        ),
    },
    {
        "patient_id": "p004",
        "note_text": (
            "Patient is a 55-year-old male with poorly controlled type 2 diabetes. "
            "HbA1c 9.4%. BMI 34.2. Currently on metformin 2000mg daily and glimepiride "
            "4mg daily. Reports frequent episodes of blurred vision and increased thirst. "
            "Fundoscopic exam reveals mild non-proliferative diabetic retinopathy. "
            "Adding empagliflozin 10mg daily. Ophthalmology referral placed. "
            "Dietary counseling reinforced."
        ),
    },
    {
        "patient_id": "p004",
        "note_text": (
            "Lab results: Fasting glucose 210 mg/dL. Creatinine 0.9 mg/dL, eGFR 92. "
            "Urine albumin-to-creatinine ratio 45 mg/g (mildly elevated, A2 category). "
            "Liver function tests normal. Potassium 4.2 mEq/L. "
            "No contraindications to empagliflozin. Patient educated on signs of "
            "genital infections and ketoacidosis."
        ),
    },
    {
        "patient_id": "p005",
        "note_text": (
            "Patient is a 38-year-old female presenting with joint pain, fatigue, and "
            "a butterfly-shaped facial rash. ANA positive (titer 1:640, homogeneous). "
            "Anti-dsDNA elevated at 120 IU/mL. Complement C3 low at 65 mg/dL (normal "
            "90-180). CBC shows mild leukopenia (WBC 3200). Urinalysis: proteinuria "
            "2+. Assessment: systemic lupus erythematosus with possible renal involvement. "
            "Starting hydroxychloroquine 200mg twice daily. Nephrology consult requested."
        ),
    },
    {
        "patient_id": "p005",
        "note_text": (
            "Renal biopsy results: Class III lupus nephritis (focal proliferative). "
            "Creatinine stable at 1.1 mg/dL. 24-hour urine protein 1.2g. "
            "Initiated mycophenolate mofetil 1000mg twice daily in addition to "
            "hydroxychloroquine. Prednisone 40mg daily with taper plan. "
            "Monitor CBC, CMP, and urinalysis every 2 weeks initially."
        ),
    },
    {
        "patient_id": "p006",
        "note_text": (
            "Patient is a 60-year-old male admitted with acute exacerbation of COPD. "
            "Presents with worsening dyspnea, increased sputum production (yellow-green), "
            "and wheezing. FEV1 38% predicted. SpO2 88% on room air, improved to 93% "
            "on 2L nasal cannula. Current home medications: tiotropium, fluticasone/"
            "salmeterol. Started on prednisone 40mg daily, nebulized albuterol and "
            "ipratropium, and azithromycin for possible bacterial trigger."
        ),
    },
    {
        "patient_id": "p006",
        "note_text": (
            "Hospital day 3: Patient improving. SpO2 95% on room air. Reduced wheezing "
            "on exam. Tolerating oral medications. Sputum culture: normal flora. "
            "Plan for discharge tomorrow with prednisone taper, continue home inhalers, "
            "and pulmonary rehabilitation referral. Smoking cessation counseling provided."
        ),
    },
    {
        "patient_id": "p007",
        "note_text": (
            "Patient is a 28-year-old female, G2P1, at 32 weeks gestation presenting "
            "with headache, visual changes, and blood pressure 160/105 mmHg. Urine "
            "dipstick 3+ protein. Platelets 142,000. AST 52 U/L. Diagnosis: preeclampsia "
            "with severe features. Administered magnesium sulfate for seizure prophylaxis. "
            "Betamethasone given for fetal lung maturity. Monitoring in L&D with plan "
            "for delivery if condition worsens."
        ),
    },
    {
        "patient_id": "p007",
        "note_text": (
            "Update: Blood pressure stabilized on IV labetalol. 24-hour urine protein "
            "3.8g. Fetal heart tracing reassuring. Repeat labs: platelets 138,000, "
            "AST 48 U/L, LDH 280 U/L. No evidence of HELLP syndrome. Plan to continue "
            "expectant management with daily labs and fetal monitoring. Target delivery "
            "at 34 weeks if stable."
        ),
    },
    {
        "patient_id": "p008",
        "note_text": (
            "Patient is a 50-year-old male with chronic hepatitis C, genotype 1a. "
            "Liver biopsy shows stage F3 fibrosis (bridging fibrosis). ALT 78 U/L, "
            "AST 65 U/L. Viral load 2.1 million IU/mL. No prior treatment. "
            "Starting sofosbuvir/velpatasvir (Epclusa) 400/100mg daily for 12 weeks. "
            "Baseline CBC, CMP, and HCV RNA obtained. No drug-drug interactions "
            "with current medications (omeprazole, ibuprofen PRN)."
        ),
    },
    {
        "patient_id": "p008",
        "note_text": (
            "Week 4 of HCV treatment: Patient tolerating Epclusa well. Reports mild "
            "fatigue. ALT improved to 42 U/L. HCV RNA now undetectable. "
            "Advised to avoid alcohol and hepatotoxic medications. "
            "Continue treatment for full 12-week course. SVR12 check planned "
            "at 12 weeks post-treatment completion."
        ),
    },
    {
        "patient_id": "p009",
        "note_text": (
            "Patient is a 65-year-old female with newly diagnosed stage IIIA non-small "
            "cell lung cancer (adenocarcinoma). PET-CT shows right upper lobe mass 4.2cm "
            "with ipsilateral mediastinal lymph node involvement. Brain MRI negative for "
            "metastases. EGFR mutation negative. ALK negative. PD-L1 TPS 60%. "
            "Recommend concurrent chemoradiation with carboplatin/paclitaxel followed "
            "by durvalumab consolidation. Pulmonary function adequate for treatment."
        ),
    },
    {
        "patient_id": "p009",
        "note_text": (
            "Oncology follow-up: Completed 4 cycles of carboplatin/paclitaxel with "
            "concurrent radiation. CT chest shows partial response with 40% reduction "
            "in tumor size. No new lesions. Manageable side effects: grade 1 nausea, "
            "grade 2 fatigue, mild esophagitis. Transitioning to durvalumab 10mg/kg "
            "every 2 weeks for consolidation. Monitor for immune-related adverse events."
        ),
    },
    {
        "patient_id": "p010",
        "note_text": (
            "Patient is a 42-year-old male presenting to the ED with severe epigastric "
            "pain radiating to the back, nausea, and vomiting. Lipase 1450 U/L (normal "
            "<60). History of heavy alcohol use. CT abdomen shows peripancreatic fat "
            "stranding consistent with acute pancreatitis. No necrosis or pseudocyst. "
            "NPO, IV fluids (lactated Ringer's at 250 mL/hr), pain management with "
            "IV hydromorphone. Monitoring for organ failure."
        ),
    },
    {
        "patient_id": "p010",
        "note_text": (
            "Hospital day 4: Pain improving with oral analgesics. Lipase trending down "
            "to 320 U/L. Tolerating clear liquid diet. No signs of organ failure "
            "(no tachycardia, creatinine normal, PaO2 normal). Transitioning to low-fat "
            "solid diet. Plan for discharge with outpatient follow-up. "
            "Alcohol cessation counseling and GI follow-up arranged."
        ),
    },
]


# -----------------
# Mock Patient Records
# -----------------
MOCK_PATIENTS = [
    {"patient_id": "p001", "first_name": "Robert", "last_name": "Chen", "gender": "male", "race": "Asian", "ethnicity": "Non-Hispanic", "marital_status": "Married", "city": "Boston", "state": "MA", "country": "US", "age": 67},
    {"patient_id": "p002", "first_name": "Sarah", "last_name": "Williams", "gender": "female", "race": "White", "ethnicity": "Non-Hispanic", "marital_status": "Single", "city": "Chicago", "state": "IL", "country": "US", "age": 45},
    {"patient_id": "p003", "first_name": "Eleanor", "last_name": "Martinez", "gender": "female", "race": "White", "ethnicity": "Hispanic", "marital_status": "Widowed", "city": "Miami", "state": "FL", "country": "US", "age": 72},
    {"patient_id": "p004", "first_name": "James", "last_name": "Thompson", "gender": "male", "race": "Black", "ethnicity": "Non-Hispanic", "marital_status": "Married", "city": "Atlanta", "state": "GA", "country": "US", "age": 55},
    {"patient_id": "p005", "first_name": "Maria", "last_name": "Garcia", "gender": "female", "race": "White", "ethnicity": "Hispanic", "marital_status": "Single", "city": "Houston", "state": "TX", "country": "US", "age": 38},
    {"patient_id": "p006", "first_name": "William", "last_name": "Johnson", "gender": "male", "race": "White", "ethnicity": "Non-Hispanic", "marital_status": "Divorced", "city": "Denver", "state": "CO", "country": "US", "age": 60},
    {"patient_id": "p007", "first_name": "Emily", "last_name": "Brown", "gender": "female", "race": "White", "ethnicity": "Non-Hispanic", "marital_status": "Married", "city": "Seattle", "state": "WA", "country": "US", "age": 28},
    {"patient_id": "p008", "first_name": "David", "last_name": "Lee", "gender": "male", "race": "Asian", "ethnicity": "Non-Hispanic", "marital_status": "Married", "city": "San Francisco", "state": "CA", "country": "US", "age": 50},
    {"patient_id": "p009", "first_name": "Patricia", "last_name": "Davis", "gender": "female", "race": "Black", "ethnicity": "Non-Hispanic", "marital_status": "Widowed", "city": "Philadelphia", "state": "PA", "country": "US", "age": 65},
    {"patient_id": "p010", "first_name": "Michael", "last_name": "Wilson", "gender": "male", "race": "White", "ethnicity": "Non-Hispanic", "marital_status": "Single", "city": "Phoenix", "state": "AZ", "country": "US", "age": 42},
]


# -----------------
# Mock Conditions, Labs, Medications
# -----------------
MOCK_CONDITIONS = [
    {"patient_id": "p001", "condition": "Essential hypertension", "code": "59621000", "system": "SNOMED", "onset_date": "2020-03-15", "abatement_date": "", "clinical_status": "active"},
    {"patient_id": "p001", "condition": "Diabetes mellitus type 2", "code": "44054006", "system": "SNOMED", "onset_date": "2018-07-22", "abatement_date": "", "clinical_status": "active"},
    {"patient_id": "p002", "condition": "Community-acquired pneumonia", "code": "385093006", "system": "SNOMED", "onset_date": "2025-04-10", "abatement_date": "2025-04-20", "clinical_status": "resolved"},
    {"patient_id": "p003", "condition": "Alzheimer disease", "code": "26929004", "system": "SNOMED", "onset_date": "2024-10-05", "abatement_date": "", "clinical_status": "active"},
    {"patient_id": "p004", "condition": "Diabetes mellitus type 2", "code": "44054006", "system": "SNOMED", "onset_date": "2015-01-18", "abatement_date": "", "clinical_status": "active"},
    {"patient_id": "p004", "condition": "Non-proliferative diabetic retinopathy", "code": "390834004", "system": "SNOMED", "onset_date": "2025-02-10", "abatement_date": "", "clinical_status": "active"},
    {"patient_id": "p005", "condition": "Systemic lupus erythematosus", "code": "55464009", "system": "SNOMED", "onset_date": "2025-01-20", "abatement_date": "", "clinical_status": "active"},
    {"patient_id": "p005", "condition": "Lupus nephritis", "code": "68815009", "system": "SNOMED", "onset_date": "2025-03-01", "abatement_date": "", "clinical_status": "active"},
    {"patient_id": "p006", "condition": "Chronic obstructive pulmonary disease", "code": "13645005", "system": "SNOMED", "onset_date": "2019-06-12", "abatement_date": "", "clinical_status": "active"},
    {"patient_id": "p009", "condition": "Non-small cell lung cancer", "code": "254637007", "system": "SNOMED", "onset_date": "2025-01-15", "abatement_date": "", "clinical_status": "active"},
    {"patient_id": "p010", "condition": "Acute pancreatitis", "code": "197456007", "system": "SNOMED", "onset_date": "2025-04-01", "abatement_date": "2025-04-05", "clinical_status": "resolved"},
]

MOCK_LABS = [
    {"patient_id": "p001", "observation": "Hemoglobin A1c", "code": "4548-4", "system": "LOINC", "value": 8.2, "unit": "%", "date": "2025-04-01", "category": "laboratory"},
    {"patient_id": "p001", "observation": "Glucose [Mass/volume] in Serum or Plasma", "code": "2345-7", "system": "LOINC", "value": 186.0, "unit": "mg/dL", "date": "2025-04-01", "category": "laboratory"},
    {"patient_id": "p001", "observation": "LDL Cholesterol", "code": "2089-1", "system": "LOINC", "value": 142.0, "unit": "mg/dL", "date": "2025-04-01", "category": "laboratory"},
    {"patient_id": "p001", "observation": "Creatinine", "code": "2160-0", "system": "LOINC", "value": 1.3, "unit": "mg/dL", "date": "2025-04-01", "category": "laboratory"},
    {"patient_id": "p001", "observation": "Troponin I", "code": "10839-9", "system": "LOINC", "value": 0.08, "unit": "ng/mL", "date": "2025-04-15", "category": "laboratory"},
    {"patient_id": "p004", "observation": "Hemoglobin A1c", "code": "4548-4", "system": "LOINC", "value": 9.4, "unit": "%", "date": "2025-03-20", "category": "laboratory"},
    {"patient_id": "p004", "observation": "Glucose [Mass/volume] in Serum or Plasma", "code": "2345-7", "system": "LOINC", "value": 210.0, "unit": "mg/dL", "date": "2025-03-20", "category": "laboratory"},
    {"patient_id": "p004", "observation": "Creatinine", "code": "2160-0", "system": "LOINC", "value": 0.9, "unit": "mg/dL", "date": "2025-03-20", "category": "laboratory"},
    {"patient_id": "p005", "observation": "Anti-dsDNA", "code": "5130-0", "system": "LOINC", "value": 120.0, "unit": "IU/mL", "date": "2025-02-01", "category": "laboratory"},
    {"patient_id": "p005", "observation": "Complement C3", "code": "4485-9", "system": "LOINC", "value": 65.0, "unit": "mg/dL", "date": "2025-02-01", "category": "laboratory"},
    {"patient_id": "p008", "observation": "Alanine aminotransferase", "code": "1742-6", "system": "LOINC", "value": 78.0, "unit": "U/L", "date": "2025-01-10", "category": "laboratory"},
    {"patient_id": "p010", "observation": "Lipase", "code": "3040-3", "system": "LOINC", "value": 1450.0, "unit": "U/L", "date": "2025-04-01", "category": "laboratory"},
]

MOCK_MEDICATIONS = [
    {"patient_id": "p001", "medication": "metformin 1000 MG", "code": "860975", "system": "RxNorm", "start_date": "2018-07-22", "end_date": "", "status": "active"},
    {"patient_id": "p001", "medication": "lisinopril 20 MG", "code": "314076", "system": "RxNorm", "start_date": "2020-03-15", "end_date": "", "status": "active"},
    {"patient_id": "p002", "medication": "azithromycin 250 MG", "code": "248656", "system": "RxNorm", "start_date": "2025-04-10", "end_date": "2025-04-15", "status": "completed"},
    {"patient_id": "p003", "medication": "donepezil 5 MG", "code": "997221", "system": "RxNorm", "start_date": "2024-10-10", "end_date": "", "status": "active"},
    {"patient_id": "p003", "medication": "amlodipine 5 MG", "code": "197361", "system": "RxNorm", "start_date": "2022-01-05", "end_date": "", "status": "active"},
    {"patient_id": "p004", "medication": "metformin 1000 MG", "code": "860975", "system": "RxNorm", "start_date": "2015-01-18", "end_date": "", "status": "active"},
    {"patient_id": "p004", "medication": "glimepiride 4 MG", "code": "310488", "system": "RxNorm", "start_date": "2020-06-01", "end_date": "", "status": "active"},
    {"patient_id": "p004", "medication": "empagliflozin 10 MG", "code": "1545653", "system": "RxNorm", "start_date": "2025-03-20", "end_date": "", "status": "active"},
    {"patient_id": "p005", "medication": "hydroxychloroquine 200 MG", "code": "979092", "system": "RxNorm", "start_date": "2025-02-01", "end_date": "", "status": "active"},
    {"patient_id": "p005", "medication": "mycophenolate mofetil 500 MG", "code": "313988", "system": "RxNorm", "start_date": "2025-03-10", "end_date": "", "status": "active"},
    {"patient_id": "p006", "medication": "tiotropium 18 MCG", "code": "2108240", "system": "RxNorm", "start_date": "2019-06-15", "end_date": "", "status": "active"},
    {"patient_id": "p009", "medication": "carboplatin 150 MG", "code": "597195", "system": "RxNorm", "start_date": "2025-02-01", "end_date": "2025-04-01", "status": "completed"},
]


def generate_note_records() -> list[dict]:
    """Build note records in the same schema as clinical_notes.csv."""
    records = []
    for index, note in enumerate(MOCK_NOTES, start=1):
        records.append({
            "note_id": f"mock_note_{index:03d}",
            "patient_id": note["patient_id"],
            "bundle_file": "mock_bundle.json",
            "resource_type": "DocumentReference",
            "resource_id": f"doc_{index:03d}",
            "note_source": "mock_annotation",
            "note_text": note["note_text"],
        })
    return records


def build_vector_index(notes_df: pd.DataFrame, model_name: str) -> None:
    """Build FAISS index and vector artifacts from mock notes."""
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        print(f"Warning: cannot build vector index — {exc}")
        print("Install with: pip install sentence-transformers faiss-cpu")
        return

    print(f"Encoding {len(notes_df)} note(s) with {model_name}...")
    model = SentenceTransformer(model_name)
    texts = notes_df["note_text"].tolist()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    dimension = int(embeddings.shape[1])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save artifacts
    np.save(os.path.join(OUTPUT_DIR, "clinical_note_embeddings.npy"), embeddings)
    faiss.write_index(index, os.path.join(OUTPUT_DIR, "clinical_note_index.faiss"))
    notes_df.to_csv(os.path.join(OUTPUT_DIR, "note_vector_metadata.csv"), index=False)

    config = {
        "model_name": model_name,
        "embedding_dimension": dimension,
        "note_count": len(notes_df),
        "metadata_file": "note_vector_metadata.csv",
        "embeddings_file": "clinical_note_embeddings.npy",
        "index_file": "clinical_note_index.faiss",
    }
    with open(os.path.join(OUTPUT_DIR, "vector_index_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved FAISS index ({index.ntotal} vectors, dim={dimension})")


def main() -> None:
    """Generate all mock datasets and vector artifacts."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_name = "all-MiniLM-L6-v2"

    # 1. Notes
    note_records = generate_note_records()
    notes_df = pd.DataFrame(note_records)
    notes_df.to_csv(os.path.join(OUTPUT_DIR, "clinical_notes.csv"), index=False)
    print(f"Generated {len(notes_df)} mock clinical notes -> clinical_notes.csv")

    # 2. Patients
    patients_df = pd.DataFrame(MOCK_PATIENTS)
    patients_df.to_csv(os.path.join(OUTPUT_DIR, "patients.csv"), index=False)
    print(f"Generated {len(patients_df)} mock patients -> patients.csv")

    # 3. Conditions
    conditions_df = pd.DataFrame(MOCK_CONDITIONS)
    conditions_df.to_csv(os.path.join(OUTPUT_DIR, "conditions.csv"), index=False)
    print(f"Generated {len(conditions_df)} mock conditions -> conditions.csv")

    # 4. Labs
    labs_df = pd.DataFrame(MOCK_LABS)
    labs_df.to_csv(os.path.join(OUTPUT_DIR, "labs.csv"), index=False)
    print(f"Generated {len(labs_df)} mock lab results -> labs.csv")

    # 5. Medications
    medications_df = pd.DataFrame(MOCK_MEDICATIONS)
    medications_df.to_csv(os.path.join(OUTPUT_DIR, "medications.csv"), index=False)
    print(f"Generated {len(medications_df)} mock medications -> medications.csv")

    # 6. Vector index
    build_vector_index(notes_df, model_name)

    print(f"\nAll mock datasets ready in '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
