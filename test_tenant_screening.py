#!/usr/bin/env python3

import os
import sys

# Add the AI Assistant directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'AI Assistant'))

from chatbot_integration import TenantScreeningHandler

def test_tenant_screening():
    print("Testing Tenant Screening Handler...")
    
    # Initialize handler
    handler = TenantScreeningHandler()
    
    # Test conversation 1: Initial request
    conversation_history = []
    user_message = "can you do tenant screening?"
    
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
    
    # Test conversation 2: Provide tenant information
    user_message = "he's 23 yo, he's employed and earns 5000 a month with credit score of 750 and has clear rental history"
    
    print(f"\n--- Test 2: Provide Tenant Information ---")
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
    
    # Test conversation 3: Add missing rent information
    user_message = "the rent is 1200"
    
    print(f"\n--- Test 3: Add Missing Rent ---")
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
    
    # Test conversation 4: Confirm and run screening
    user_message = "yes"
    
    print(f"\n--- Test 4: Confirmation ---")
    print(f"User: {user_message}")
    
    result = handler.handle(conversation_history, user_message, result['fields'])
    print(f"AI Response: {result['response']}")
    print(f"Action: {result['action']}")
    print(f"Fields: {result['fields']}")
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_tenant_screening()
