"""
Test client for Video Detection API
Usage: python test_client.py
"""

import requests
import json
import time
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n🔍 Testing health endpoint...")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_upload_video(video_path: str):
    """Test video upload and processing"""
    print(f"\n📤 Uploading video: {video_path}")
    
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return None
    
    with open(video_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_BASE_URL}/process/upload", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Upload successful!")
        print(f"Job ID: {result['job_id']}")
        print(f"Video ID: {result['video_id']}")
        print(f"Total detections: {result['results']['total_detections']}")
        print(f"Known persons: {result['results']['known_persons']}")
        print(f"Unknown persons: {result['results']['unknown_persons']}")
        return result
    else:
        print(f"❌ Upload failed: {response.status_code}")
        print(f"Error: {response.text}")
        return None

def test_process_from_url(video_url: str):
    """Test video processing from URL"""
    print(f"\n🌐 Processing video from URL: {video_url}")
    
    payload = {
        "url": video_url,
        "video_name": "test_video_from_url"
    }
    
    response = requests.post(
        f"{API_BASE_URL}/process/url",
        json=payload
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Processing successful!")
        print(f"Job ID: {result['job_id']}")
        print(f"Video ID: {result['video_id']}")
        return result
    else:
        print(f"❌ Processing failed: {response.status_code}")
        print(f"Error: {response.text}")
        return None

def test_get_results(job_id: str):
    """Test getting processing results"""
    print(f"\n📊 Getting results for job: {job_id}")
    
    response = requests.get(f"{API_BASE_URL}/results/{job_id}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Results retrieved!")
        print(f"Total detections: {result.get('total_detections', 0)}")
        print(f"Known persons: {result.get('known_persons', 0)}")
        print(f"Unknown persons: {result.get('unknown_persons', 0)}")
        return result
    else:
        print(f"❌ Failed to get results: {response.status_code}")
        return None

def test_get_detections(video_id: str, limit: int = 5):
    """Test getting detections for a video"""
    print(f"\n🎯 Getting detections for video: {video_id}")
    
    response = requests.get(
        f"{API_BASE_URL}/detections/{video_id}",
        params={"skip": 0, "limit": limit}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Retrieved {result['total']} detections")
        
        # Print first few detections
        for i, detection in enumerate(result['detections'][:3], 1):
            print(f"\nDetection {i}:")
            print(f"  Frame: {detection['frame_number']}")
            print(f"  Timestamp: {detection['timestamp']:.2f}s")
            print(f"  Confidence: {detection['confidence']:.2f}")
            print(f"  Is Known: {detection['is_known']}")
            if detection['is_known']:
                print(f"  Person: {detection['matched_person_name']}")
                print(f"  Similarity: {detection['similarity_score']:.2f}")
        
        return result
    else:
        print(f"❌ Failed to get detections: {response.status_code}")
        return None

def test_add_known_person():
    """Test adding a known person"""
    print("\n👤 Adding known person...")
    
    # Generate dummy feature vector (384 dimensions for ViT-S/16)
    import random
    feature_vector = [random.random() for _ in range(384)]
    
    payload = {
        "person_id": "test_person_001",
        "name": "Test Person",
        "feature_vector": feature_vector
    }
    
    response = requests.post(
        f"{API_BASE_URL}/features/add",
        json=payload
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Known person added!")
        print(f"Person ID: {result['person_id']}")
        return result
    else:
        print(f"❌ Failed to add person: {response.status_code}")
        print(f"Error: {response.text}")
        return None

def test_list_known_persons():
    """Test listing known persons"""
    print("\n📋 Listing known persons...")
    
    response = requests.get(f"{API_BASE_URL}/features/list")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Found {result['total']} known persons")
        
        for person in result['persons']:
            print(f"\n  - ID: {person['person_id']}")
            print(f"    Name: {person['name']}")
            print(f"    Added: {person['added_at']}")
        
        return result
    else:
        print(f"❌ Failed to list persons: {response.status_code}")
        return None

def run_all_tests():
    """Run all test cases"""
    print("=" * 60)
    print("🧪 Video Detection API Test Suite")
    print("=" * 60)
    
    # Test 1: Health check
    if not test_health():
        print("\n❌ Health check failed. Is the server running?")
        return
    
    # Test 2: Add known person
    test_add_known_person()
    time.sleep(1)
    
    # Test 3: List known persons
    test_list_known_persons()
    time.sleep(1)
    
    # Test 4: Upload and process video
    # Replace with your actual video path
    video_path = "test_video.mp4"
    
    print("\n" + "=" * 60)
    print("📹 Testing Video Processing")
    print("=" * 60)
    
    if Path(video_path).exists():
        result = test_upload_video(video_path)
        
        if result:
            job_id = result['job_id']
            video_id = result['video_id']
            
            time.sleep(2)
            
            # Test 5: Get results
            test_get_results(job_id)
            time.sleep(1)
            
            # Test 6: Get detections
            test_get_detections(video_id)
    else:
        print(f"\n⚠️  Skipping video upload test - file not found: {video_path}")
        print("   Place a test video at this path to run the full test suite")
    
    # Alternative: Test with URL
    # Uncomment and replace with actual video URL
    # video_url = "https://example.com/test_video.mp4"
    # result = test_process_from_url(video_url)
    
    print("\n" + "=" * 60)
    print("✅ Test suite completed!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()