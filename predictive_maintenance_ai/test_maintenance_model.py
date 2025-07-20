import pandas as pd
import joblib
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'models', 'maintenance_rf_model.pkl')
model = joblib.load(model_path)

# Example test input (from the new schema)
test_data = pd.DataFrame([
    {
        'address': '1800',
        'age_years': 25,
        'last_service_years_ago': 2,
        'seasonality': 'Winter'
    },
    {
        'address': '809',
        'age_years': 60,
        'last_service_years_ago': 5,
        'seasonality': 'Summer'
    }
])

# Predict risk scores
risk_scores = model.predict(test_data)

# Map risk scores to recommended actions
def get_recommended_action(score):
    if score > 7:
        return 'Immediate Action'
    elif score > 4:
        return 'Monitor'
    else:
        return 'Routine'

recommended_actions = [get_recommended_action(score) for score in risk_scores]

for i, row in test_data.iterrows():
    print(f"Address: {row['address']}, Age: {row['age_years']}, Last Service: {row['last_service_years_ago']} years ago, Seasonality: {row['seasonality']}")
    print(f"  Predicted risk score: {risk_scores[i]:.2f}, Recommended action: {recommended_actions[i]}")
