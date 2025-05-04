import requests
import json

# Base URL for the API
base_url = "http://localhost:8000/api/cars/search"

# Test 1: Search with search_query parameter only
def test_search_query():
    params = {
        "search_query": "BMW",
        "partial_match": "true"
    }
    response = requests.get(base_url, params=params)
    print(f"Test 1 - Search Query 'BMW':")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Total Results: {data.get('total', 0)}")
        print(f"First few results:")
        for car in data.get('items', [])[:3]:
            print(f"  - {car.get('brand', '')} {car.get('model', '')}")
    else:
        print(f"Error: {response.text}")
    print()

# Test 2: Search with search_query and other filters
def test_search_with_filters():
    params = {
        "search_query": "Sedan",
        "min_price": "20000",
        "partial_match": "true"
    }
    response = requests.get(base_url, params=params)
    print(f"Test 2 - Search Query 'Sedan' with min_price 20000:")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Total Results: {data.get('total', 0)}")
        print(f"First few results:")
        for car in data.get('items', [])[:3]:
            print(f"  - {car.get('brand', '')} {car.get('model', '')} - ${car.get('price', 0)}")
    else:
        print(f"Error: {response.text}")
    print()

# Run the tests
if __name__ == "__main__":
    print("Starting search API tests...\n")
    
    # Start the server first (this is just a reminder)
    print("Make sure the server is running on http://localhost:8000\n")
    
    # Run the tests
    test_search_query()
    test_search_with_filters()
    
    print("Tests completed.")
