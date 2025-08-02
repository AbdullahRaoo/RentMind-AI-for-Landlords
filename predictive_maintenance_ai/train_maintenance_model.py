import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os

# Load data
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'maintenance_model_inputs_outputs.csv')
df = pd.read_csv(DATA_PATH)

# Features and target
y = df['risk_score']
X = df[['address', 'age_years', 'last_service_years_ago', 'seasonality']]

# Preprocessing
categorical = ['address', 'seasonality']
numerical = ['age_years', 'last_service_years_ago']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', StandardScaler(), numerical)
])

# Model pipeline
model = Pipeline([
    ('pre', preprocessor),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
pred_test = model.predict(X_test)
print(f'R^2 on test set: {score:.3f}')

# Save model and preprocessor
joblib.dump(model, os.path.join(os.path.dirname(__file__), 'models', 'maintenance_rf_model.pkl'))
print('Model saved to models/maintenance_rf_model.pkl')
