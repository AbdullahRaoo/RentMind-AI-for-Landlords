import pandas as pd
import numpy as np
import xgboost as xgb

# Load the trained model
model = xgb.Booster()
model.load_model("rent_xgboost_model.json")

# Define the input features for prediction
# Replace these values with actual inputs
input_data = {
    'address': 1308,  # Encoded address
    'subdistrict_code': 182,  # Encoded subdistrict
    'BEDROOMS': 2.0,
    'BATHROOMS': 1.0,
    'SIZE': 700.0,
    'PROPERTY TYPE': 9,  # Encoded property type
    'avg_distance_to_nearest_station': 0.4,
    'nearest_station_count': 3.0
}

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Prepare the data for XGBoost
dinput = xgb.DMatrix(input_df, missing=np.nan)

# Make predictions
predicted_log_rent = model.predict(dinput)
predicted_rent = np.expm1(predicted_log_rent)  # Reverse log transformation

# Calculate rent range (±10%)
predicted_rent = predicted_rent[0]
lower_rent = int(predicted_rent - 0.10 * predicted_rent)
upper_rent = int(predicted_rent + 0.10 * predicted_rent)

# Define RMSE (use the RMSE from your training process, e.g., £1039.64)
rmse = 1039.64  # Replace with the actual RMSE from your training output

# Calculate confidence score
confidence = max(0, 1 - (rmse / predicted_rent))  # Confidence level as a percentage
confidence_percentage = round(confidence * 100, 2)

# Output the results
print(f"🎯 Predicted Rent: £{int(predicted_rent)}")
print(f"💡 Rent Range: £{lower_rent}–£{upper_rent}")
print(f"🔒 Confidence Score: {confidence_percentage}%")