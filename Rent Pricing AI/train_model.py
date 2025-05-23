import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the cleaned data
df = pd.read_csv("data/cleaned_rent_data.csv")

# Handle missing values
df['SIZE'] = df['SIZE'].fillna(df['SIZE'].median())
df['BATHROOMS'] = df['BATHROOMS'].fillna(df['BATHROOMS'].median())

# Log-transform the target variable
df['log_rent'] = np.log1p(df['rent'])  # Log transformation
target = 'log_rent'

# Features and target
features = [
    'address', 'subdistrict_code', 'BEDROOMS', 'BATHROOMS', 'SIZE',
    'PROPERTY TYPE', 'avg_distance_to_nearest_station', 'nearest_station_count'
]

X = df[features]
y = df[target]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Prepare data for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)
dtest = xgb.DMatrix(X_test, label=y_test, missing=np.nan)

# Model parameters
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'learning_rate': 0.1,
    'seed': 42
}

# Train the model
model = xgb.train(
    params,
    dtrain,
    num_boost_round=200,
    evals=[(dtest, 'test')],
    early_stopping_rounds=10,
    verbose_eval=10
)

# Predict and evaluate
y_pred_log = model.predict(dtest)
y_pred = np.expm1(y_pred_log)  # Reverse log transformation
rmse = np.sqrt(mean_squared_error(np.expm1(y_test), y_pred))  # Use original scale
print(f"\n✅ Test RMSE: £{rmse:.2f}\n")

# Show a few predictions with confidence ranges and explanations
print("🎯 Suggested Rent Ranges with Confidence and Explanations:")
for i in range(5):  # Show first 5 predictions
    pred = y_pred[i]
    lower = int(pred - 0.10 * pred)
    upper = int(pred + 0.10 * pred)
    confidence = max(0, 1 - (rmse / pred))  # Confidence level as a percentage
    confidence_percentage = round(confidence * 100, 2)

    print(f"\nPrediction #{i+1}:")
    print(f" - Estimated Rent: £{int(pred)}")
    print(f" - Rent Range: £{lower}–£{upper}")
    print(f" - Confidence Level: {confidence_percentage}%")

    # Explanation based on features
    x_sample = X_test.iloc[i]
    print(" - Explanation:")
    for col in features:
        print(f"   - {col}: {x_sample[col]}")

# Feature importance
importance = model.get_score(importance_type='weight')
sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

print("\n🧠 Top Features Driving Rent Prediction:")
for feat, score in sorted_features:
    print(f" - {feat}: importance score {score}")

# Save the model
model.save_model("rent_xgboost_model.json")
print("\n💾 Model saved to 'rent_xgboost_model.json'")
