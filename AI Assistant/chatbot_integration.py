import os
import pandas as pd
import numpy as np
import xgboost as xgb
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Dynamically construct the absolute path to the model file
current_dir = os.path.dirname(__file__)  # Directory of the current script
model_path = os.path.join(current_dir, "../Rent Pricing AI/rent_xgboost_model.json")

# Load the trained model
model = xgb.Booster()
model.load_model(model_path)

def predict_rent(input_data):
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

    return predicted_rent, lower_rent, upper_rent, confidence_percentage

def generate_explanation_with_langchain(input_data, predicted_rent, lower_rent, upper_rent, confidence_percentage):
    # Initialize LangChain's ChatOpenAI
    chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_api_key)

    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=[
            "address", "subdistrict_code", "bedrooms", "bathrooms", "size",
            "property_type", "avg_distance", "station_count", "predicted_rent",
            "lower_rent", "upper_rent", "confidence_percentage"
        ],
        template="""
        Based on the following property details:
        - Address (encoded): {address}
        - Subdistrict Code: {subdistrict_code}
        - Bedrooms: {bedrooms}
        - Bathrooms: {bathrooms}
        - Size: {size} sq ft
        - Property Type (encoded): {property_type}
        - Average Distance to Nearest Station: {avg_distance} miles
        - Nearest Station Count: {station_count}

        The predicted rent is £{predicted_rent}, with a range of £{lower_rent}–£{upper_rent}.
        The confidence level of this prediction is {confidence_percentage}%.

        Please provide a user-friendly explanation of this prediction.
        """
    )

    # Format the prompt with input data
    prompt = prompt_template.format(
        address=input_data['address'],
        subdistrict_code=input_data['subdistrict_code'],
        bedrooms=input_data['BEDROOMS'],
        bathrooms=input_data['BATHROOMS'],
        size=input_data['SIZE'],
        property_type=input_data['PROPERTY TYPE'],
        avg_distance=input_data['avg_distance_to_nearest_station'],
        station_count=input_data['nearest_station_count'],
        predicted_rent=int(predicted_rent),
        lower_rent=lower_rent,
        upper_rent=upper_rent,
        confidence_percentage=confidence_percentage
    )

    # Generate explanation using LangChain
    response = chat.invoke([HumanMessage(content=prompt)])
    explanation = response.content.strip()
    return explanation

# Example input data
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

# Predict rent
predicted_rent, lower_rent, upper_rent, confidence_percentage = predict_rent(input_data)

# Generate explanation
explanation = generate_explanation_with_langchain(input_data, predicted_rent, lower_rent, upper_rent, confidence_percentage)

# Output the results
print(f"🎯 Predicted Rent: £{int(predicted_rent)}")
print(f"💡 Rent Range: £{lower_rent}–£{upper_rent}")
print(f"🔒 Confidence Score: {confidence_percentage}%")
print(f"📝 Explanation: {explanation}")