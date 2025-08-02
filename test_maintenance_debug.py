#!/usr/bin/env python3
"""
Debug script to test maintenance prediction locally
"""

import os
import sys
import traceback

# Add the AI Assistant directory to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'AI Assistant'))

def test_maintenance_model():
    """Test the maintenance prediction model directly"""
    print("🔧 Testing Maintenance Prediction Model...")
    print("=" * 50)
    
    try:
        # Import the handler
        from chatbot_integration import MaintenancePredictionHandler
        
        # Create handler instance
        handler = MaintenancePredictionHandler()
        print("✅ Handler created successfully")
        
        # Test model loading
        try:
            model = handler.get_model()
            print("✅ Model loaded successfully")
            print(f"   Model type: {type(model)}")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            traceback.print_exc()
            return
        
        # Test address map loading
        try:
            address_map = handler.get_address_map()
            print("✅ Address map loaded successfully")
            print(f"   Address map size: {len(address_map)} entries")
            print(f"   Sample addresses: {list(address_map.keys())[:5]}")
        except Exception as e:
            print(f"❌ Address map loading failed: {e}")
            traceback.print_exc()
            return
        
        # Test field extraction
        print("\n🧠 Testing Field Extraction...")
        test_message = "Property at Holland Road, 50 years old, last serviced 5 years ago, winter season"
        fields = handler.extract_fields(test_message, [], {})
        print(f"✅ Extracted fields: {fields}")
        
        # Test field encoding
        print("\n🔢 Testing Field Encoding...")
        encoded = handler.encode_fields_for_model(fields)
        print(f"✅ Encoded fields: {encoded}")
        
        # Test model prediction
        print("\n🎯 Testing Model Prediction...")
        test_fields = {
            'address': 'Holland Road',
            'age_years': 50,
            'last_service_years_ago': 5,
            'seasonality': 'winter'
        }
        
        result = handler.run_model(test_fields)
        print("✅ Model prediction successful!")
        print(f"Result preview: {result[:200]}...")
        
        print("\n" + "=" * 50)
        print("✅ All maintenance prediction tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()

def test_conversation_flow():
    """Test the full conversation flow"""
    print("\n🗣️ Testing Conversation Flow...")
    print("=" * 50)
    
    try:
        from chatbot_integration import MaintenancePredictionHandler
        
        handler = MaintenancePredictionHandler()
        conversation_history = []
        
        # Test message 1: Initial request
        msg1 = "I need maintenance prediction for my property"
        result1 = handler.handle(conversation_history, msg1, {})
        print(f"✅ Message 1 handled: {result1['action']}")
        
        # Update conversation history
        conversation_history.append({"role": "user", "content": msg1})
        conversation_history.append({"role": "assistant", "content": result1['response']})
        
        # Test message 2: Provide details
        msg2 = "Holland Road property, 50 years old, last service 5 years ago, winter season"
        result2 = handler.handle(conversation_history, msg2, result1['fields'])
        print(f"✅ Message 2 handled: {result2['action']}")
        print(f"   Fields collected: {list(result2['fields'].keys())}")
        
        # Update conversation history
        conversation_history.append({"role": "user", "content": msg2})
        conversation_history.append({"role": "assistant", "content": result2['response']})
        
        # Test message 3: Confirmation
        msg3 = "yes"
        result3 = handler.handle(conversation_history, msg3, result2['fields'])
        print(f"✅ Message 3 handled: {result3['action']}")
        
        if result3['action'] == 'maintenance_prediction':
            print("✅ Full conversation flow successful!")
            print(f"Final result preview: {result3['response'][:200]}...")
        else:
            print(f"⚠️ Expected maintenance_prediction action, got: {result3['action']}")
            
    except Exception as e:
        print(f"❌ Conversation flow test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_maintenance_model()
    test_conversation_flow()
