#!/usr/bin/env python
"""
Script to test that car specification fields are properly populated
"""
import asyncio
import sys
import os
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the car service
from app.services.car_service import car_service

async def test_car_retrieval():
    """Test retrieving a car with specifications"""
    print("Testing car retrieval with specifications...")
    
    # Try to get the Audi allroad quattro (ID: 7663)
    car_id = 7663
    car = await car_service.get_car_by_id(car_id)
    
    if car:
        print(f"Successfully retrieved car: {car['name']}")
        print("Car data:")
        print(json.dumps(car, indent=2))
        
        # Check if specifications are populated
        if car['specifications']:
            print("\nSpecifications are present!")
            print("Engine HP:", car['specifications'].get('engine_hp'))
            print("Engine Cylinders:", car['specifications'].get('engine_cylinders'))
            print("Engine Fuel Type:", car['specifications'].get('engine_fuel_type'))
            print("Transmission Type:", car['specifications'].get('transmission_type'))
        else:
            print("\nSpecifications are NULL!")
    else:
        print(f"Car with ID {car_id} not found")

if __name__ == "__main__":
    asyncio.run(test_car_retrieval()) 