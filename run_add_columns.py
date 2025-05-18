#!/usr/bin/env python
"""
Script to add missing columns to the car_specifications table
"""
import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the add columns function
from app.db.add_spec_columns import main

if __name__ == "__main__":
    print("Starting column addition...")
    asyncio.run(main())
    print("Column addition completed.") 