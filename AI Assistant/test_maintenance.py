#!/usr/bin/env python3
"""
Test the improved maintenance prediction handler
"""

import sys
sys.path.append('d:/work/tallal-projects/RentMind-AI-for-Landlords/AI Assistant')

from chatbot_integration import handle_conversation

def test_maintenance_prediction():
    print("🔧 Testing Improved Maintenance Prediction Handler")
    print("=" * 60)
    
    # Test 1: Initial request for maintenance prediction
    print("\n--- Test 1: Initial Request ---")
    result1 = handle_conversation([], "can you do maintenance prediction?")
    print(f"User: can you do maintenance prediction?")
    print(f"Response: {result1.get('response', 'No response')[:200]}...")
    print(f"Action: {result1.get('action', 'unknown')}")
    
    # Test 2: Providing partial information
    print("\n--- Test 2: Providing Partial Info ---")
    conversation_history = [
        {"role": "user", "content": "can you do maintenance prediction?"},
        {"role": "assistant", "content": result1.get('response', '')}
    ]
    
    result2 = handle_conversation(
        conversation_history, 
        "the property is at 123 Oak Street, it's 10 years old",
        last_candidate_fields=result1.get('fields', {}),
        last_intent=result1.get('last_intent'),
        intent_completed=result1.get('intent_completed', False)
    )
    print(f"User: the property is at 123 Oak Street, it's 10 years old")
    print(f"Response: {result2.get('response', 'No response')[:200]}...")
    print(f"Action: {result2.get('action', 'unknown')}")
    print(f"Fields collected: {result2.get('fields', {})}")
    
    # Test 3: Providing more information
    print("\n--- Test 3: More Information ---")
    conversation_history.extend([
        {"role": "user", "content": "the property is at 123 Oak Street, it's 10 years old"},
        {"role": "assistant", "content": result2.get('response', '')}
    ])
    
    result3 = handle_conversation(
        conversation_history,
        "last serviced 2 years ago, and it's winter now",
        last_candidate_fields=result2.get('fields', {}),
        last_intent=result2.get('last_intent'),
        intent_completed=result2.get('intent_completed', False)
    )
    print(f"User: last serviced 2 years ago, and it's winter now")
    print(f"Response: {result3.get('response', 'No response')[:200]}...")
    print(f"Action: {result3.get('action', 'unknown')}")
    print(f"Fields collected: {result3.get('fields', {})}")
    
    # Test 4: Confirmation
    print("\n--- Test 4: Confirmation ---")
    if result3.get('fields', {}) and all(field in result3['fields'] and result3['fields'][field] not in (None, '', 0, 0.0) for field in ['address', 'age_years', 'last_service_years_ago', 'seasonality']):
        conversation_history.extend([
            {"role": "user", "content": "last serviced 2 years ago, and it's winter now"},
            {"role": "assistant", "content": result3.get('response', '')}
        ])
        
        result4 = handle_conversation(
            conversation_history,
            "yes, that's correct",
            last_candidate_fields=result3.get('fields', {}),
            last_intent=result3.get('last_intent'),
            intent_completed=result3.get('intent_completed', False)
        )
        print(f"User: yes, that's correct")
        print(f"Response: {result4.get('response', 'No response')[:200]}...")
        print(f"Action: {result4.get('action', 'unknown')}")
    else:
        print("❌ Not enough fields collected to test confirmation")
    
    print("\n" + "=" * 60)
    print("✅ Maintenance prediction testing complete!")

if __name__ == "__main__":
    test_maintenance_prediction()
