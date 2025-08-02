import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

# --- Load Rent Pricing Properties (for address sync) ---
rent_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Rent Pricing AI/data/cleaned_rent_data.csv'))
rent_df = pd.read_csv(rent_data_path)


# Use every row in rent data (not just unique addresses)
property_rows = rent_df.reset_index(drop=True)

# --- 1. Property Metadata ---
properties = []
cities = ['London', 'Manchester', 'Birmingham', 'Leeds', 'Bristol']
construction_types = ['Victorian Brick', 'Post-War Concrete', 'Modern Timber Frame', 'Edwardian Stone']

for idx, row in property_rows.iterrows():
    pid = row['address']
    age = np.random.randint(5, 120)
    properties.append({
        'property_id': pid,
        'address': str(pid),  # Use address exactly as in rent data
        'age_years': age,
        'construction_type': random.choice(construction_types),
        'last_service_years_ago': np.random.randint(0, 10),
        'seasonality': random.choice(['Winter', 'Spring', 'Summer', 'Autumn'])
    })

df_properties = pd.DataFrame(properties)

# --- 2. Maintenance Logs (for training) ---
logs = []
for idx, row in property_rows.iterrows():
    pid = row['address']
    # Ensure at least one log per property row
    for _ in range(1, np.random.randint(2, 4)):
        logs.append({
            'property_id': pid,
            'date': (datetime.now() - timedelta(days=np.random.randint(30, 365*3))).strftime('%Y-%m-%d'),
            'issue_type': random.choice(['Boiler Failure', 'Damp/Mould', 'Roof Leak', 'Electrical Fault']),
            'severity': random.choice(['Low', 'Medium', 'High']),
            'resolved': random.choice([True, False])
        })
df_logs = pd.DataFrame(logs)

# --- 3. Tenant Reports (for training) ---
reports = []
for idx, row in property_rows.iterrows():
    pid = row['address']
    # Ensure at least one report per property row
    for _ in range(1, np.random.randint(2, 4)):
        reports.append({
            'property_id': pid,
            'date': (datetime.now() - timedelta(days=np.random.randint(1, 365))).strftime('%Y-%m-%d'),
            'issue_reported': random.choice(['No Heating', 'Damp Patch', 'Leaking Tap', 'Broken Lock']),
            'urgency': np.random.randint(1, 6)
        })
df_reports = pd.DataFrame(reports)

# --- 4. Seasonality (for training) ---
seasonal = []
for idx, row in property_rows.iterrows():
    pid = row['address']
    for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
        seasonal.append({
            'property_id': pid,
            'season': season,
            'avg_temp_c': np.random.randint(2, 22),
            'rainfall_mm': round(np.random.uniform(0.5, 5.0), 1),
            'maintenance_risk_score': np.random.randint(1, 10)
        })
df_seasonal = pd.DataFrame(seasonal)

# --- 5. Model Input/Output Example (for inference) ---
# Input: address, age, last_service_years_ago, seasonality
# Output: risk_score, recommended_action
model_inputs = df_properties[['address', 'age_years', 'last_service_years_ago', 'seasonality']].copy()
model_inputs['risk_score'] = np.random.randint(1, 10, size=len(model_inputs))
model_inputs['recommended_action'] = model_inputs['risk_score'].apply(lambda x: 'Immediate Action' if x > 7 else ('Monitor' if x > 4 else 'Routine'))

# Ensure all output files have exactly the same number of rows as rent data
assert len(model_inputs) == len(property_rows) == 3478, f"Expected 3478 rows, got {len(model_inputs)}"
# --- Export to CSV ---
data_dir = os.path.join(os.path.dirname(__file__), 'data/')
os.makedirs(data_dir, exist_ok=True)
df_properties.to_csv(os.path.join(data_dir, 'uk_properties.csv'), index=False)
df_logs.to_csv(os.path.join(data_dir, 'uk_maintenance_logs.csv'), index=False)
df_reports.to_csv(os.path.join(data_dir, 'uk_tenant_reports.csv'), index=False)
df_seasonal.to_csv(os.path.join(data_dir, 'uk_seasonal_risks.csv'), index=False)
model_inputs.to_csv(os.path.join(data_dir, 'maintenance_model_inputs_outputs.csv'), index=False)
