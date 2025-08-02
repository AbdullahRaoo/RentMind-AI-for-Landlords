import sys
import os
import json
import numpy as np
# Add both possible paths for local development and Docker
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../AI Assistant')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../AI_Assistant')))
from chatbot_integration import predict_rent

def to_native(val):
    if isinstance(val, (np.generic, np.floating, np.integer)):
        # Convert numpy scalars to Python types
        if isinstance(val, np.floating):
            return float(val)
        if isinstance(val, np.integer):
            return int(val)
        return val.item()
    if hasattr(val, 'item'):
        return val.item()
    if isinstance(val, (list, tuple)):
        return [to_native(v) for v in val]
    if isinstance(val, dict):
        return {k: to_native(v) for k, v in val.items()}
    if isinstance(val, (float, int, str, bool)):
        return val
    try:
        return float(val)
    except Exception:
        return str(val)

def process_property_message(input_data):
    # Convert all input_data values to native types before passing to predict_rent
    input_data_native = {k: to_native(v) for k, v in input_data.items()}
    predicted_rent, lower_rent, upper_rent, confidence_percentage = predict_rent(input_data_native)
    # Convert all outputs to native types
    return {
        'predicted_rent': int(float(to_native(predicted_rent))),
        'lower_rent': int(float(to_native(lower_rent))),
        'upper_rent': int(float(to_native(upper_rent))),
        'confidence_percentage': round(float(to_native(confidence_percentage)), 2),
    }
