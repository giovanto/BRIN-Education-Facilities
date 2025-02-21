"""
Local Nominatim connection test and debug script
"""
import requests
from geopy.geocoders import Nominatim
import logging
import time
from typing import Optional, Dict

def test_nominatim_connection(host: str = "localhost", port: int = 8080) -> bool:
    """
    Test connection to local Nominatim instance
    """
    try:
        # Test basic connection
        url = f"http://{host}:{port}/status.php"
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {str(e)}")
        return False

def test_reverse_geocode(
    lat: float = 52.3676,
    lon: float = 4.9041,
    host: str = "localhost",
    port: int = 8080
) -> Optional[Dict]:
    """
    Test reverse geocoding with local Nominatim
    """
    url = f"http://{host}:{port}/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        "addressdetails": 1
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request error: {str(e)}")
        return None

def main():
    """
    Test local Nominatim connection and functionality
    """
    # Configuration - adjust these values
    host = "localhost"  # or your Nominatim host
    port = 8080        # or your Nominatim port
    
    print("Testing Nominatim connection...")
    
    # Test basic connection
    if test_nominatim_connection(host, port):
        print("✓ Basic connection successful")
    else:
        print("✗ Connection failed")
        print("Please check:")
        print("1. Is Nominatim running?")
        print("2. Correct host and port?")
        print("3. Firewall settings?")
        return
    
    # Test reverse geocoding
    print("\nTesting reverse geocoding...")
    test_coords = [
        (52.3676, 4.9041, "Amsterdam test"),
        (51.9244, 4.4777, "Rotterdam test")
    ]
    
    for lat, lon, desc in test_coords:
        print(f"\nTesting {desc}...")
        result = test_reverse_geocode(lat, lon, host, port)
        if result:
            print(f"✓ Success: {result.get('display_name', 'No display name')}")
        else:
            print(f"✗ Failed for coordinates {lat}, {lon}")
            
    print("\nTesting rate limits...")
    start_time = time.time()
    for i in range(5):
        result = test_reverse_geocode(52.3676, 4.9041, host, port)
        if result:
            print(f"✓ Request {i+1} successful")
        else:
            print(f"✗ Request {i+1} failed")
        time.sleep(0.1)  # 100ms delay
    end_time = time.time()
    print(f"Time for 5 requests: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()