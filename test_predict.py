import requests
import time
import sys

# Configuration
BASE_URL = "http://localhost:8000"
PREDICT_ENDPOINT = f"{BASE_URL}/predict"

def test_predict_endpoint():
    """Automatic test for the /predict endpoint"""
    print("🧪 Testing /predict endpoint...")
    print("=" * 50)
    
    # Test cases with different feature combinations
    test_cases = [
        {
            "name": "High Performance Student",
            "features": {
                "student_id": 1,
                "hours_studied": 30.0,
                "sleep_hours": 9.0,
                "attendance_percent": 0.95,
                "previous_scores": 85
            },
            "expected_range": (10, 100)
        },
        {
            "name": "Average Student",
            "features": {
                "student_id": 1,
                "hours_studied": 6.0,
                "sleep_hours": 8.0,
                "attendance_percent": 0.95,
                "previous_scores": 85
            },
            "expected_range": (20, 85)
        },
        {
            "name": "Low Performance Student",
            "features": {
                "student_id": 1,
                "hours_studied": 1.0,
                "sleep_hours": 4.0,
                "attendance_percent": 0.5,
                "previous_scores": 55
            },
            "expected_range": (1, 50)
        }
    ]
    
    # Check if service is running
    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ Service is not healthy")
            return False
    except requests.exceptions.RequestException:
        print("❌ Service is not running. Please start the service first.")
        return False
    
    print("✅ Service is running and healthy")
    
    # Run test cases
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['name']} ---")
        
        # Prepare request
        request_data = {
            "features": test_case["features"]
        }
        
        try:
            # Make prediction request
            start_time = time.time()
            response = requests.post(
                PREDICT_ENDPOINT,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            end_time = time.time()
            
            # Check response status
            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                model_info = result["model_info"]
                
                print(f"✅ Request successful")
                print(f"📊 Prediction: {prediction}")
                print(f"⏱️  Response time: {(end_time - start_time)*1000:.2f}ms")
                print(f"🤖 Model: {model_info.get('model_name', 'Unknown')}")
                print(f"📄 Features used: {test_case['features']}")
                
                # Validate prediction range (if applicable)
                if isinstance(prediction, (int, float)):
                    min_expected, max_expected = test_case["expected_range"]
                    if min_expected <= prediction <= max_expected:
                        print(f"✅ Prediction within expected range [{min_expected}, {max_expected}]")
                    else:
                        print(f"⚠️  Prediction outside expected range [{min_expected}, {max_expected}]")
                
            else:
                print(f"❌ Request failed with status {response.status_code}")
                print(f"📝 Error: {response.text}")
                all_passed = False
                
        except requests.exceptions.Timeout:
            print("❌ Request timed out")
            all_passed = False
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {str(e)}")
            all_passed = False
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            all_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests PASSED!")
        print("✅ /predict endpoint is working correctly")
    else:
        print("❌ Some tests FAILED!")
        print("🔍 Please check the service logs and model")
    
    return all_passed

def performance_test():
    """Test prediction endpoint performance"""
    print("\n🚀 Performance Testing...")
    print("=" * 50)
    
    test_features = {
        "student_id": 1,
        "hours_studied": 6.0,
        "sleep_hours": 8.0,
        "attendance_percent": 0.95,
        "previous_scores": 85
    }
    
    request_data = {"features": test_features}
    num_requests = 10
    response_times = []
    
    for i in range(num_requests):
        try:
            start_time = time.time()
            response = requests.post(PREDICT_ENDPOINT, json=request_data, timeout=5)
            end_time = time.time()
            
            if response.status_code == 200:
                response_times.append((end_time - start_time) * 1000)
            else:
                print(f"❌ Request {i+1} failed")
                
        except Exception as e:
            print(f"❌ Request {i+1} error: {str(e)}")
    
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        print(f"📊 Performance Results ({len(response_times)}/{num_requests} successful):")
        print(f"   Average: {avg_time:.2f}ms")
        print(f"   Minimum: {min_time:.2f}ms")
        print(f"   Maximum: {max_time:.2f}ms")
    else:
        print("❌ No successful requests for performance testing")

if __name__ == "__main__":
    success = test_predict_endpoint()
    performance_test()
    
    sys.exit(0 if success else 1)
