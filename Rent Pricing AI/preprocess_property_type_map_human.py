import pandas as pd
import json

# Load the CSV file
input_csv = "data/rent_ads_rightmove_extended.csv"
df = pd.read_csv(input_csv)

# Create property type code-to-label mapping BEFORE encoding
ptype_cats = df['PROPERTY TYPE'].astype('category')
ptype_map_human = {int(code): cat for code, cat in enumerate(ptype_cats.cat.categories)}

# Save the mapping to a new JSON file
with open("property_type_map_human.json", "w", encoding="utf-8") as f:
    json.dump(ptype_map_human, f, ensure_ascii=False, indent=2)

print(f"Saved human-readable property type map to property_type_map_human.json with {len(ptype_map_human)} entries.")
