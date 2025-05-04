import pytest
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app
from app.api.csv_data import convert_row_to_car_response, load_csv_data
import os

client = TestClient(app)

# Sample test data
sample_row = pd.Series({
    "Make": "Test",
    "Model": "Car",
    "Year": 2022,
    "Engine Fuel Type": "gasoline",
    "Engine HP": 200,
    "Engine Cylinders": 4,
    "Transmission Type": "AUTOMATIC",
    "Driven_Wheels": "all wheel drive",
    "Number of Doors": 4,
    "Market Category": "Luxury",
    "Vehicle Size": "Midsize",
    "Vehicle Style": "Sedan",
    "highway MPG": 30.0,
    "city mpg": 25,
    "Popularity": 1000,
    "MSRP": 35000.0
})

# Test the convert_row_to_car_response function
def test_convert_row_to_car_response():
    # Test with valid data
    car_dict = convert_row_to_car_response(sample_row, car_id=1)
    
    # Check basic fields
    assert car_dict["id"] == 1
    assert car_dict["name"] == "Test Car"
    assert car_dict["brand"] == "Test"
    assert car_dict["model"] == "Car"
    assert car_dict["year"] == 2022
    
    # Check detailed fields
    assert car_dict["engine_fuel_type"] == "gasoline"
    assert car_dict["engine_hp"] == 200
    assert car_dict["engine_cylinders"] == 4
    assert car_dict["transmission_type"] == "AUTOMATIC"
    assert car_dict["driven_wheels"] == "all wheel drive"
    assert car_dict["number_of_doors"] == 4
    assert car_dict["market_category"] == "Luxury"
    assert car_dict["vehicle_size"] == "Midsize"
    assert car_dict["vehicle_style"] == "Sedan"
    assert car_dict["highway_mpg"] == 30.0
    assert car_dict["city_mpg"] == 25
    assert car_dict["popularity"] == 1000
    assert car_dict["msrp"] == 35000.0
    
    # Check specifications object
    assert car_dict["specifications"] is not None
    assert car_dict["specifications"]["engine"] == "200 HP, 4 cylinders"
    assert car_dict["specifications"]["transmission"] == "AUTOMATIC"
    assert car_dict["specifications"]["fuel_type"] == "gasoline"
    assert car_dict["specifications"]["mileage"] == 30.0
    assert car_dict["specifications"]["seating_capacity"] == 5
    assert car_dict["specifications"]["body_type"] == "Sedan"

# Test with invalid data
def test_convert_row_with_invalid_data():
    # Missing required fields
    invalid_row = pd.Series({
        "Model": "Car",
        "Year": 2022
    })
    
    car_dict = convert_row_to_car_response(invalid_row, car_id=1)
    
    # Should return a valid car object with default values
    assert car_dict["id"] == 1
    assert car_dict["brand"] == "Unknown"
    assert car_dict["model"] == "Car"
    assert car_dict["specifications"] is not None

# Test with out-of-range values
def test_convert_row_with_out_of_range_values():
    # Out of range values
    invalid_row = pd.Series({
        "Make": "Test",
        "Model": "Car",
        "Year": 2022,
        "Engine HP": 5000,  # Too high
        "Engine Cylinders": 20,  # Too high
        "Number of Doors": 10,  # Too high
        "highway MPG": 200.0,  # Too high
        "city mpg": 180,  # Too high
        "MSRP": -1000.0  # Negative
    })
    
    car_dict = convert_row_to_car_response(invalid_row, car_id=1)
    
    # Should use default values for out-of-range fields
    assert car_dict["engine_hp"] == 0
    assert car_dict["engine_cylinders"] == 0
    assert car_dict["number_of_doors"] == 4
    assert car_dict["highway_mpg"] == 0.0
    assert car_dict["city_mpg"] == 0
    assert car_dict["msrp"] == 0.0

# Test API endpoints
def test_get_car_by_id():
    # Skip if CSV file doesn't exist in test environment
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cars_data.csv")
    if not os.path.exists(csv_path):
        pytest.skip("CSV file not found, skipping test")
    
    response = client.get("/api/csv-data/cars/1")
    assert response.status_code == 200
    
    car_data = response.json()
    assert car_data["id"] == 1
    assert "specifications" in car_data
    assert car_data["specifications"] is not None

def test_get_car_invalid_id():
    response = client.get("/api/csv-data/cars/999999")  # Invalid ID
    assert response.status_code == 404

def test_search_cars():
    # Skip if CSV file doesn't exist in test environment
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cars_data.csv")
    if not os.path.exists(csv_path):
        pytest.skip("CSV file not found, skipping test")
    
    # Test search by make
    response = client.get("/api/csv-data/cars/search?make=BMW")
    assert response.status_code == 200
    
    cars = response.json()
    assert len(cars) > 0
    assert all(car["brand"] == "BMW" for car in cars)
    
    # Test search by vehicle_style
    response = client.get("/api/csv-data/cars/search?vehicle_style=Sedan")
    assert response.status_code == 200
    
    cars = response.json()
    assert len(cars) > 0
    assert all(car["vehicle_style"] == "Sedan" for car in cars)
