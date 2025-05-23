import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv("rent_ads_rightmove_extended.csv")

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
