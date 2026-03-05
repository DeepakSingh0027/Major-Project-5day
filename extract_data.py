import json
import os
import pandas as pd

data_dir = "synthea/output/fhir"   # folder containing patient JSON bundles



print("\nDatasets saved successfully!")

patients = []
conditions = []
observations = []
medications = []

files = os.listdir(data_dir)

for file in files:

    if not file.endswith(".json"):
        continue

    path = os.path.join(data_dir, file)

    with open(path, encoding="utf-8", errors="ignore") as f:
        bundle = json.load(f)

    patient_id = None

    for entry in bundle["entry"]:

        resource = entry["resource"]
        rtype = resource["resourceType"]

        # -----------------
        # Patient
        # -----------------
        if rtype == "Patient":

            patient_id = resource.get("id")
            gender = resource.get("gender")
            birth = resource.get("birthDate")

            patients.append({
                "patient_id": patient_id,
                "gender": gender,
                "birthDate": birth
            })

        # -----------------
        # Condition
        # -----------------
        elif rtype == "Condition":

            if "code" in resource:
                name = resource["code"]["coding"][0].get("display")

                conditions.append({
                    "patient_id": patient_id,
                    "condition": name
                })

        # -----------------
        # Observation
        # -----------------
        elif rtype == "Observation":

            if "valueQuantity" in resource:

                value = resource["valueQuantity"].get("value")
                unit = resource["valueQuantity"].get("unit")

                observations.append({
                    "patient_id": patient_id,
                    "value": value,
                    "unit": unit
                })

        # -----------------
        # Medication
        # -----------------
        elif rtype == "MedicationRequest":

            med = None

            if "medicationCodeableConcept" in resource:
                med = resource["medicationCodeableConcept"]["coding"][0].get("display")

            elif "medicationReference" in resource:
                med = resource["medicationReference"].get("reference")

            if med:
                medications.append({
                    "patient_id": patient_id,
                    "medication": med
                })


print("Extraction complete\n")

print("Patients:", len(patients))
print("Conditions:", len(conditions))
print("Observations:", len(observations))
print("Medications:", len(medications))

# Convert lists to DataFrames
patients_df = pd.DataFrame(patients)
conditions_df = pd.DataFrame(conditions)
medications_df = pd.DataFrame(medications)
observations_df = pd.DataFrame(observations)

# Save CSV files
patients_df.to_csv("datasets/patients.csv", index=False)
conditions_df.to_csv("datasets/conditions.csv", index=False)
medications_df.to_csv("datasets/medications.csv", index=False)
observations_df.to_csv("datasets/observations.csv", index=False)