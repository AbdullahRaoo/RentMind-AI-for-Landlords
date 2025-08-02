#!/usr/bin/env python3
"""
Comprehensive test script for all RentMind AI features
Tests maintenance prediction, rent estimation, and tenant screening
"""

import requests
import json
import time
import sys
from datetime import datetime

# Configuration
BASE_URL = "https://rentmind.srv889806.hstgr.cloud"
WEBSOCKET_URL = "wss://rentmind.srv889806.hstgr.cloud/ws/chat/"

def test_maintenance_prediction():
    """Test maintenance prediction feature"""
    print("\n🔧 Testing Maintenance Prediction...")
    
    # Test data for maintenance prediction
    maintenance_data = {
        "message": "I need to predict maintenance for my property",
        "property_id": "test_prop_123",
        "property_age": 15,
        "building_type": "apartment",
        "last_maintenance": "2023-06-15",
        "hvac_age": 8,
        "plumbing_age": 12,
        "electrical_age": 10,
        "roof_age": 5,
        "previous_issues": ["plumbing", "hvac"]
    }
    
    try:
        # Using HTTP request to test the chatbot endpoint
        response = requests.post(
            f"{BASE_URL}/api/chat/",
            json=maintenance_data,
            timeout=30,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Maintenance prediction successful")
            print(f"   Response: {result.get('response', 'No response')[:100]}...")
            return True
        else:
            print(f"❌ Maintenance prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Maintenance prediction error: {str(e)}")
        return False

def test_rent_estimation():
    """Test rent estimation feature"""
    print("\n💰 Testing Rent Estimation...")
    
    # Test data for rent estimation
    rent_data = {
        "message": "I need to estimate rent for my property",
        "property_type": "apartment",
        "bedrooms": 2,
        "bathrooms": 1,
        "area": 850,
        "location": "Downtown Dubai",
        "amenities": ["parking", "gym", "pool"],
        "building_age": 5
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/chat/",
            json=rent_data,
            timeout=30,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Rent estimation successful")
            print(f"   Response: {result.get('response', 'No response')[:100]}...")
            return True
        else:
            print(f"❌ Rent estimation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Rent estimation error: {str(e)}")
        return False

def test_tenant_screening():
    """Test tenant screening feature"""
    print("\n👥 Testing Tenant Screening...")
    
    # Test data for tenant screening
    tenant_data = {
        "message": "I need to screen a potential tenant",
        "tenant_name": "John Doe",
        "income": 8000,
        "employment_status": "employed",
        "credit_score": 750,
        "rental_history": "good",
        "references": 3,
        "background_check": "clean"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/chat/",
            json=tenant_data,
            timeout=30,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Tenant screening successful")
            print(f"   Response: {result.get('response', 'No response')[:100]}...")
            return True
        else:
            print(f"❌ Tenant screening failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Tenant screening error: {str(e)}")
        return False

def test_general_chat():
    """Test general chatbot functionality"""
    print("\n💬 Testing General Chat...")
    
    # Test general chat
    chat_data = {
        "message": "Hello, can you help me with my rental property?"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/chat/",
            json=chat_data,
            timeout=30,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ General chat successful")
            print(f"   Response: {result.get('response', 'No response')[:100]}...")
            return True
        else:
            print(f"❌ General chat failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ General chat error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting RentMind AI Features Test Suite")
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 Base URL: {BASE_URL}")
    
    # Test website accessibility
    print("\n🌍 Testing Website Accessibility...")
    try:
        response = requests.get(BASE_URL, timeout=10)
        if response.status_code == 200:
            print("✅ Website is accessible")
        else:
            print(f"❌ Website accessibility failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Website accessibility error: {str(e)}")
        return False
    
    # Run all AI feature tests
    tests = [
        ("General Chat", test_general_chat),
        ("Maintenance Prediction", test_maintenance_prediction),
        ("Rent Estimation", test_rent_estimation),
        ("Tenant Screening", test_tenant_screening)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {str(e)}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All AI features are working correctly!")
        return True
    else:
        print("⚠️  Some features need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
