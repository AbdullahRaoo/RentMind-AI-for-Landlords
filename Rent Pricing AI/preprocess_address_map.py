import pandas as pd
import json

# Load the CSV file
input_csv = "data/rent_ads_rightmove_extended.csv"
df = pd.read_csv(input_csv)

# Create address-to-code mapping BEFORE encoding
address_cats = df['address'].astype('category')
address_map = {cat: code for code, cat in enumerate(address_cats.cat.categories)}

# Save the mapping to a new JSON file
with open("address_map_human.json", "w", encoding="utf-8") as f:
    json.dump(address_map, f, ensure_ascii=False, indent=2)

print(f"Saved human-readable address map to address_map_human.json with {len(address_map)} entries.")
