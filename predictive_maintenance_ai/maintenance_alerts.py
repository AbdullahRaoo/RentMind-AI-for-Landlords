# In your maintenance_alerts.py
import pandas as pd
import joblib
import os

# Load CORRECT preprocessor and model
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.pkl'))
clf = joblib.load(os.path.join(MODEL_DIR, 'xgb_model.pkl'))

# Load your data (add this before feature_cols and X)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
properties = pd.read_csv(os.path.join(DATA_DIR, 'uk_properties.csv'))
logs = pd.read_csv(os.path.join(DATA_DIR, 'uk_maintenance_logs.csv'))
reports = pd.read_csv(os.path.join(DATA_DIR, 'uk_tenant_reports.csv'))
seasonal = pd.read_csv(os.path.join(DATA_DIR, 'uk_seasonal_risks.csv'))

# Feature engineering (same as in preprocessing)
logs['is_urgent'] = logs['severity'].apply(lambda x: 1 if x in ['High', 'Critical'] else 0)
reports['is_open'] = reports['status'].apply(lambda x: 1 if x != 'Resolved' else 0)
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
df = df.fillna(0)

# Define the SAME features used in training
feature_cols = [
    'construction_type',
    'age_years',
    'hvac_age',
    'plumbing_age',
    'roof_age',
    'total_incidents',
    'urgent_incidents',
    'open_issues'
]

# Process and predict
X = df[feature_cols].copy()
X_processed = preprocessor.transform(X)
risk_scores = clf.predict(X_processed)
df['predicted_risk_score'] = risk_scores

# Map risk scores to action labels
bins = [0, 4.5, 7.5, 10]
action_labels = [
    'Routine monitoring',
    'Schedule maintenance soon',
    'Immediate inspection and preventive maintenance required'
]
df['predicted_action'] = pd.cut(df['predicted_risk_score'], bins=bins, labels=action_labels, include_lowest=True)

# Print prediction distribution for debugging
print('Prediction distribution:')
print(df['predicted_action'].value_counts())

# Generate alerts
alert_actions = [
    'Immediate inspection and preventive maintenance required',
    'Schedule maintenance soon'
]
alerts = df[df['predicted_action'].isin(alert_actions)]

# Print alerts
for _, row in alerts.iterrows():
    print(f"\n🔔 [Maintenance Alert] {row['address']}")
    print(f"Recommended Action: {row['predicted_action']}")
    print(f"Risk Factors: Age {row['age_years']} yrs, Open Issues {int(row['open_issues'])}, Urgent Incidents {int(row['urgent_incidents'])}")
    print(f"Last Inspection: {row['last_inspection_date']}")
    print("---")

print(f"\nTotal alerts generated: {len(alerts)}")