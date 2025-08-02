#!/usr/bin/env python3
"""
Test script to simulate maintenance prediction calls via the API
"""

import requests
import json
import time

def test_maintenance_prediction():
    """Test the maintenance prediction API endpoint"""
    
    base_url = "https://rentmind.hstgr.cloud"
    
    # Test data for maintenance prediction
    test_messages = [
        {
            "type": "text", 
            "message": "I need maintenance prediction for my property"
        },
        {
            "type": "text", 
            "message": "located at Holland Road and it's age is 50 years old and it's been 5 years since last maintenance and it's winter now"
        },
        {
            "type": "text", 
            "message": "yes"
        }
    ]
    
    print("🔧 Testing Maintenance Prediction API...")
    print("=" * 50)
    
    session = requests.Session()
    
    for i, msg_data in enumerate(test_messages, 1):
        print(f"\n--- Message {i} ---")
        print(f"Sending: {msg_data['message']}")
        
        try:
            response = session.post(
                f"{base_url}/chat/",
                json=msg_data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "TestScript/1.0"
                },
                timeout=30
            )
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Response: {result.get('response', 'No response')[:200]}...")
                print(f"Action: {result.get('action', 'Unknown')}")
                
                if 'error' in result.get('action', ''):
                    print("⚠️ Error detected in response!")
                    break
                    
            else:
                print(f"Error: HTTP {response.status_code}")
                print(f"Response: {response.text[:200]}...")
                break
                
        except requests.exceptions.Timeout:
            print("❌ Request timed out - likely a hang/crash!")
            break
        except requests.exceptions.ConnectionError:
            print("❌ Connection error - server may have crashed!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            break
            
        # Wait between requests
        time.sleep(2)
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    test_maintenance_prediction()
