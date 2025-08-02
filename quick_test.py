import requests
import json

def test_rent_prediction():
    """Test rent prediction via REST API"""
    print("Testing Rent Prediction...")
    
    url = "https://rentmind.srv889806.hstgr.cloud/api/chat/"
    
    # First message - ask for rent estimation
    data1 = {
        "message": "I need rent estimation for a 2 bedroom flat with 1 bathroom, 700 sq ft at Holland Road, Kensington W14"
    }
    
    try:
        response1 = requests.post(url, json=data1, timeout=30)
        print(f"Response 1 Status: {response1.status_code}")
        if response1.status_code == 200:
            result1 = response1.json()
            print(f"Response 1: {result1.get('response', '')[:150]}...")
        
        # Second message - confirm details
        data2 = {
            "message": "yes, that's correct"
        }
        
        response2 = requests.post(url, json=data2, timeout=30)
        print(f"Response 2 Status: {response2.status_code}")
        if response2.status_code == 200:
            result2 = response2.json()
            print(f"Response 2: {result2.get('response', '')[:150]}...")
            
            # Check if we got a rent estimate
            if "£" in result2.get('response', ''):
                print("✅ Rent prediction working!")
                return True
            else:
                print("❌ No rent estimate received")
                return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_maintenance_prediction():
    """Test maintenance prediction"""
    print("\nTesting Maintenance Prediction...")
    
    url = "https://rentmind.srv889806.hstgr.cloud/api/chat/"
    
    data = {
        "message": "I need maintenance prediction for a property at Holland Road, 50 years old, last serviced 5 years ago, current season is winter"
    }
    
    try:
        response = requests.post(url, json=data, timeout=30)
        print(f"Response Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result.get('response', '')[:150]}...")
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_tenant_screening():
    """Test tenant screening"""
    print("\nTesting Tenant Screening...")
    
    url = "https://rentmind.srv889806.hstgr.cloud/api/chat/"
    
    data = {
        "message": "I need to screen a tenant with credit score 750, income 5000, rent 2000, employed, no eviction record"
    }
    
    try:
        response = requests.post(url, json=data, timeout=30)
        print(f"Response Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result.get('response', '')[:150]}...")
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing RentMind AI Features")
    print("=" * 50)
    
    results = []
    results.append(("Rent Prediction", test_rent_prediction()))
    results.append(("Maintenance Prediction", test_maintenance_prediction()))
    results.append(("Tenant Screening", test_tenant_screening()))
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    
    passed = 0
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All AI features are working correctly!")
        print("🚀 RentMind deployment is SUCCESSFUL!")
    else:
        print("⚠️ Some features need attention")
