# Preprocessing and Feature Engineering for Predictive Maintenance AI
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import joblib

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load data from data subfolder
data_dir = 'data/'
properties = pd.read_csv(os.path.join(data_dir, 'uk_properties.csv'))
logs = pd.read_csv(os.path.join(data_dir, 'uk_maintenance_logs.csv'))
reports = pd.read_csv(os.path.join(data_dir, 'uk_tenant_reports.csv'))
seasonal = pd.read_csv(os.path.join(data_dir, 'uk_seasonal_risks.csv'))

# Feature engineering
logs['is_urgent'] = logs['severity'].apply(lambda x: 1 if x in ['High', 'Critical'] else 0)
reports['is_open'] = reports['status'].apply(lambda x: 1 if x != 'Resolved' else 0)

# Merge datasets (example: focus on June for seasonal risk)
df = properties.merge(
    logs.groupby('property_id').agg(
        total_incidents=('log_id', 'count'),
        urgent_incidents=('is_urgent', 'sum')
    ), on='property_id', how='left'
).merge(
    reports.groupby('property_id').agg(
        open_issues=('is_open', 'sum')
    ), on='property_id', how='left'
).merge(
    seasonal[seasonal['month'] == 'June'][['property_id', 'maintenance_risk_score']],
    on='property_id', how='left'
)

# Fill missing values (if any)
df = df.fillna(0)

# Define the correct feature columns (excluding metadata)
feature_cols = [
    'construction_type',  # categorical
    'age_years',         # numeric
    'hvac_age',          # numeric  
    'plumbing_age',      # numeric
    'roof_age',          # numeric
    'total_incidents',   # numeric
    'urgent_incidents',  # numeric
    'open_issues'        # numeric
]

# Create preprocessor that only uses the correct features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['construction_type']),
        ('num', StandardScaler(), [col for col in feature_cols if col != 'construction_type'])
    ],
    remainder='drop'  # This ensures ONLY specified columns are processed
)

# Process features
X = df[feature_cols]
y = df['maintenance_risk_score']  # Target variable
X_processed = preprocessor.fit_transform(X)

# Get feature names after transformation
cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(['construction_type'])
num_features = [col for col in feature_cols if col != 'construction_type']
feature_names = list(cat_features) + num_features

# Verify shape
print(f"Transformed X shape: {X_processed.shape}, expected features: {len(feature_names)}")

# Save the full merged dataset with all original columns
df.to_csv(os.path.join(data_dir, 'preprocessed_full_data.csv'), index=False)

# Save the processed features and targets
pd.DataFrame(X_processed, columns=feature_names).to_csv(
    os.path.join(data_dir, 'preprocessed_features.csv'), index=False)
pd.DataFrame(y, columns=['maintenance_risk_score']).to_csv(
    os.path.join(data_dir, 'preprocessed_targets.csv'), index=False)

# Save the preprocessor for later use
joblib.dump(preprocessor, os.path.join('models', 'preprocessor.pkl'))

print('Preprocessing complete. Files saved:')
print(f"- Full dataset: data/preprocessed_full_data.csv")
print(f"- Processed features: data/preprocessed_features.csv")
print(f"- Targets: data/preprocessed_targets.csv")
print(f"- Preprocessor: models/preprocessor.pkl")