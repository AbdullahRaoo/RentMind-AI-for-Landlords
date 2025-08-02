#!/usr/bin/env python3
"""
Simple test to find the correct API endpoint and test maintenance prediction.
"""

import requests
import json

# Base URL
BASE_URL = "https://srv889806.hstgr.cloud"

def find_correct_api_endpoint():
    """Try different possible API endpoints."""
    
    possible_endpoints = [
        "/api/chat/",
        "/api/chat",
        "/chat/",
        "/chat",
        "/backend/api/chat/",
        "/backend/api/chat",
    ]
    
    print("🔍 Searching for correct API endpoint...")
    
    for endpoint in possible_endpoints:
        url = BASE_URL + endpoint
        print(f"Trying: {url}")
        
        try:
            # Try a simple POST request
            response = requests.post(
                url,
                json={"message": "test"},
                headers={"Content-Type": "application/json"},
                timeout=10,
                allow_redirects=False  # Don't follow redirects to avoid loops
            )
            
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"✅ Found working endpoint: {url}")
                return url
            elif response.status_code in [301, 302, 307, 308]:
                print(f"  Redirect to: {response.headers.get('Location', 'Unknown')}")
            elif response.status_code == 404:
                print("  Not found")
            elif response.status_code == 405:
                print("  Method not allowed (but endpoint exists)")
                return url  # Endpoint exists, just wrong method
            else:
                print(f"  Other status: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print("  Timeout")
        except requests.exceptions.ConnectionError as e:
            print(f"  Connection error: {e}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("❌ No working API endpoint found")
    return None

def test_with_browser_simulation():
    """Try to simulate what a browser would do."""
    
    print("\n🌐 Trying browser simulation...")
    
    # First, get the main page to establish session
    try:
        session = requests.Session()
        main_response = session.get(BASE_URL, timeout=10)
        print(f"Main page status: {main_response.status_code}")
        
        # Look for any form action or API references in the HTML
        if "api" in main_response.text.lower():
            print("Found 'api' references in the page")
        if "chat" in main_response.text.lower():
            print("Found 'chat' references in the page")
            
        # Try common Django patterns
        api_url = BASE_URL + "/api/chat/"
        
        # Get CSRF token if needed
        csrf_token = None
        if "csrftoken" in main_response.text:
            import re
            csrf_match = re.search(r'csrftoken["\']?\s*[:=]\s*["\']([^"\']+)', main_response.text)
            if csrf_match:
                csrf_token = csrf_match.group(1)
                print(f"Found CSRF token: {csrf_token[:10]}...")
        
        # Prepare headers like a browser
        headers = {
            "Content-Type": "application/json",
            "Referer": BASE_URL,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        if csrf_token:
            headers["X-CSRFToken"] = csrf_token
        
        # Test the maintenance prediction
        test_payload = {
            "message": "I need maintenance prediction for my property"
        }
        
        print(f"Testing with payload: {test_payload}")
        
        response = session.post(
            api_url,
            json=test_payload,
            headers=headers,
            timeout=15
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print("✅ SUCCESS! API is working")
                print(f"Response: {result}")
                return True
            except json.JSONDecodeError:
                print("Response is not JSON")
                print(f"Content: {response.text[:500]}")
        else:
            print(f"Response content: {response.text[:500]}")
            
    except Exception as e:
        print(f"Error: {e}")
        
    return False

if __name__ == "__main__":
    print("🚀 API Endpoint Discovery and Test")
    print("=" * 50)
    
    # Try to find the correct endpoint
    endpoint = find_correct_api_endpoint()
    
    if not endpoint:
        # Try browser simulation
        test_with_browser_simulation()
