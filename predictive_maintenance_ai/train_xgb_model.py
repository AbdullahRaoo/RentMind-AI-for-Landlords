import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load preprocessed data
try:
    df = pd.read_csv('data/preprocessed_full_data.csv')
except FileNotFoundError:
    print("Error: Please run data_preprocessing.py first to generate the data file")
    exit()

# Define features and target
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
target_col = 'maintenance_risk_score'

# Prepare data
X = df[feature_cols]
y = df[target_col]

# Create preprocessor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['construction_type']),
        ('num', StandardScaler(), [col for col in feature_cols if col != 'construction_type'])
    ],
    remainder='drop'
)

# Transform features
X_processed = preprocessor.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Train model
model = XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model trained with MSE: {mse:.2f}")

# Save model and preprocessor
joblib.dump(preprocessor, 'models/preprocessor.pkl')
joblib.dump(model, 'models/xgb_model.pkl')
print("Model and preprocessor saved to models/ directory")