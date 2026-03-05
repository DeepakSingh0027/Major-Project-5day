import json

file = "synthea/output/fhir/Zaida719_Schiller186_50d7ca6b-16ee-609b-096c-ecfb288135d6.json"

with open(file) as f:
    data = json.load(f)

print("Resource types inside bundle:\n")

for entry in data["entry"]:
    resource = entry["resource"]
    print(resource["resourceType"])