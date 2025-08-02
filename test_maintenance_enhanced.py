#!/usr/bin/env python3

import os
import sys

# Add the AI Assistant directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'AI Assistant'))

from chatbot_integration import MaintenancePredictionHandler

def test_maintenance_prediction():
    print("Testing Enhanced Maintenance Prediction Handler...")
    
    # Initialize handler
    handler = MaintenancePredictionHandler()
    
    # Test conversation: Provide maintenance information (like from user's logs)
    conversation_history = []
    user_message = "can you do maintenance prediction?"
    
    print(f"\n--- Test 1: Initial Request ---")
    print(f"User: {user_message}")
    
    result = handler.handle(conversation_history, user_message)
    print(f"AI Response: {result['response']}")
    print(f"Action: {result['action']}")
    print(f"Fields: {result['fields']}")
    
    # Update conversation history
    conversation_history.extend([
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": result['response']}
    ])
    
    # Test conversation 2: Provide maintenance information (from user's logs)
    user_message = "located at Holland Road, Kensington being around 48 years old and it's been 5 years since last maintenance was carried out and it's winter now"
    
    print(f"\n--- Test 2: Provide Maintenance Information ---")
    print(f"User: {user_message}")
    
    result = handler.handle(conversation_history, user_message, result['fields'])
    print(f"AI Response: {result['response']}")
    print(f"Action: {result['action']}")
    print(f"Fields: {result['fields']}")
    
    # Update conversation history
    conversation_history.extend([
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": result['response']}
    ])
    
    # Test conversation 3: Confirm and run prediction
    user_message = "yes"
    
    print(f"\n--- Test 3: Confirmation ---")
    print(f"User: {user_message}")
    
    result = handler.handle(conversation_history, user_message, result['fields'])
    print(f"AI Response: {result['response']}")
    print(f"Action: {result['action']}")
    print(f"Fields: {result['fields']}")
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_maintenance_prediction()
