import os
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_get_car_by_id():
    # Skip if CSV file doesn't exist in test environment
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cars_data.csv")
    if not os.path.exists(csv_path):
        pytest.skip("CSV file not found, skipping test")

    response = client.get("/api/cars/1")
    assert response.status_code == 200

    car_data = response.json()
    assert car_data["id"] == 1
    assert "specifications" in car_data
    assert car_data["specifications"] is not None

def test_get_car_invalid_id():
    response = client.get("/api/cars/999999")  # Invalid ID
    assert response.status_code == 404

def test_search_cars():
    # Skip if CSV file doesn't exist in test environment
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cars_data.csv")
    if not os.path.exists(csv_path):
        pytest.skip("CSV file not found, skipping test")

    # Test search by make
    response = client.get("/api/cars/search?make=BMW")
    assert response.status_code == 200

    data = response.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data
    assert "total_pages" in data

    # Check that all items have the correct make
    for car in data["items"]:
        assert "BMW" in car["brand"]

def test_get_all_cars():
    # Skip if CSV file doesn't exist in test environment
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cars_data.csv")
    if not os.path.exists(csv_path):
        pytest.skip("CSV file not found, skipping test")

    response = client.get("/api/cars/all")
    assert response.status_code == 200

    cars = response.json()
    assert isinstance(cars, list)
    assert len(cars) > 0

    # Check that each car has the required fields
    for car in cars:
        assert "id" in car
        assert "brand" in car
        assert "model" in car
        assert "year" in car
        assert "specifications" in car
