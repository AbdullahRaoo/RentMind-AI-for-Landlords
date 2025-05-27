import pandas as pd
import numpy as np
import json

# Load the CSV file
df = pd.read_csv("data/rent_ads_rightmove_extended.csv")

# Clean 'SIZE' column: remove 'sq ft', commas, handle 'Ask agent' etc.
def clean_size(val):
    if isinstance(val, str) and 'sq ft' in val:
        val = val.replace('sq ft', '').replace(',', '').strip()
        try:
            return float(val)
        except ValueError:
            return np.nan
    return np.nan

df['SIZE'] = df['SIZE'].apply(clean_size)

# Drop rows with missing target (rent)
df = df.dropna(subset=['rent'])

# Encode categorical features
df['PROPERTY TYPE'] = df['PROPERTY TYPE'].astype('category').cat.codes
df['subdistrict_code'] = df['subdistrict_code'].astype('category').cat.codes
df['address'] = df['address'].astype('category').cat.codes

# Save mapping dictionaries for use in the backend
address_map = {cat: code for code, cat in enumerate(df['address'].astype('category').cat.categories)}
property_type_map = {cat: code for code, cat in enumerate(df['PROPERTY TYPE'].astype('category').cat.categories)}
subdistrict_code_map = {cat: code for code, cat in enumerate(df['subdistrict_code'].astype('category').cat.categories)}

with open("address_map.json", "w") as f:
    json.dump(address_map, f)
with open("property_type_map.json", "w") as f:
    json.dump(property_type_map, f)
with open("subdistrict_code_map.json", "w") as f:
    json.dump(subdistrict_code_map, f)

# Select final features and target
features = [
    'address', 'subdistrict_code', 'BEDROOMS', 'BATHROOMS', 'SIZE',
    'PROPERTY TYPE', 'avg_distance_to_nearest_station', 'nearest_station_count'
]
target = 'rent'

# Final cleaned DataFrame
df_model = df[features + [target]]

# (Optional) Save to new CSV
df_model.to_csv("cleaned_rent_data.csv", index=False)
