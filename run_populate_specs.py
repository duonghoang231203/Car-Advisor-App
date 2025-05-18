#!/usr/bin/env python
"""
Script to run the car specification population
"""
import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the populate function
from app.db.populate_car_specs import main

if __name__ == "__main__":
    print("Starting car specification population...")
    asyncio.run(main())
    print("Car specification population completed.") 