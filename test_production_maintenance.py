#!/usr/bin/env python3
"""
Test script for maintenance prediction on production server.
Tests the exact flow that was causing disconnections.
"""

import requests
import json
import time

# Production server URL
BASE_URL = "https://srv889806.hstgr.cloud"
CHAT_ENDPOINT = f"{BASE_URL}/api/chat/"

def test_maintenance_prediction_flow():
    """Test the complete maintenance prediction flow that was causing crashes."""
    
    print("🔍 Testing Maintenance Prediction Flow on Production Server")
    print(f"Server: {BASE_URL}")
    print("=" * 60)
    
    # Test data that mimics the user's typical input
    test_messages = [
        {
            "step": "1. Initial Request", 
            "message": "I need maintenance prediction for my property"
        },
        {
            "step": "2. Property Details",
            "message": "The property is at Holland Road, it's 50 years old, last serviced 5 years ago, and it's winter season"
        },
        {
            "step": "3. Confirmation (This was causing disconnections)",
            "message": "yes"
        }
    ]
    
    conversation_history = []
    
    for i, test in enumerate(test_messages, 1):
        print(f"\n{test['step']}")
        print(f"Message: '{test['message']}'")
        print("-" * 40)
        
        try:
            # Prepare request payload
            payload = {
                "message": test["message"],
                "conversation_history": conversation_history
            }
            
            print(f"Sending request to {CHAT_ENDPOINT}")
            
            # Send request with timeout
            response = requests.post(
                CHAT_ENDPOINT,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30  # 30 second timeout
            )
            
            print(f"Response Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    
                    # Display response details
                    print(f"✅ SUCCESS - Step {i} completed")
                    print(f"Response: {result.get('response', 'No response')[:200]}...")
                    print(f"Action: {result.get('action', 'No action')}")
                    print(f"Fields: {result.get('fields', {})}")
                    
                    # Update conversation history
                    conversation_history.append({
                        "role": "user",
                        "content": test["message"]
                    })
                    conversation_history.append({
                        "role": "assistant", 
                        "content": result.get('response', '')
                    })
                    
                    # Check if this was the maintenance prediction result
                    if result.get('action') == 'maintenance_prediction':
                        print("🎉 MAINTENANCE PREDICTION COMPLETED SUCCESSFULLY!")
                        print("The disconnection issue appears to be RESOLVED!")
                        break
                    elif result.get('action') == 'error':
                        print("⚠️  Error action detected, but server didn't crash")
                        print("This is an improvement - graceful error handling working")
                        break
                        
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error: {e}")
                    print(f"Raw response: {response.text[:500]}")
                    break
            else:
                print(f"❌ HTTP Error {response.status_code}")
                print(f"Response: {response.text[:500]}")
                break
                
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Connection Error: {e}")
            print("Server may have crashed or is not responding")
            break
        except requests.exceptions.Timeout as e:
            print(f"⏰ Timeout Error: {e}")
            print("Request took longer than 30 seconds")
            break
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")
            break
        
        # Small delay between requests
        time.sleep(1)
    
    print("\n" + "=" * 60)
    print("Test Complete")

def test_server_health():
    """Test if the server is responsive."""
    print("🏥 Testing Server Health")
    print("-" * 30)
    
    try:
        # Test the main page
        response = requests.get(BASE_URL, timeout=10)
        print(f"Main page status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Server is responsive")
            return True
        else:
            print(f"⚠️  Server returned status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Server health check failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Production Maintenance Prediction Test")
    print(f"Testing server: {BASE_URL}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # First check server health
    if test_server_health():
        print("\n")
        test_maintenance_prediction_flow()
    else:
        print("❌ Skipping API tests due to server issues")
