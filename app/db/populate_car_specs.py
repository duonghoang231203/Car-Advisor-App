"""
Script to populate missing car specification fields in the database.
This script will update all existing car specifications by:
1. Extracting hp and cylinders from engine field
2. Copying values from existing fields to new fields
3. Extracting vehicle_type from description
4. Setting defaults for any remaining null fields
"""

import asyncio
import re
import logging
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import mysql
from app.db.models import Car, CarSpecification

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def populate_car_specifications():
    """Populate missing car specification fields for all cars"""
    logger.info("Starting to populate car specifications...")
    
    async for session in mysql.get_session():
        try:
            # Get all cars with their specifications
            result = await session.execute(
                select(Car, CarSpecification)
                .join(CarSpecification, Car.id == CarSpecification.car_id, isouter=True)
            )
            cars_with_specs = result.fetchall()
            
            logger.info(f"Found {len(cars_with_specs)} cars to process")
            
            for car, spec in cars_with_specs:
                if not spec:
                    # Create a new specification if one doesn't exist
                    logger.info(f"Creating new specification for car ID {car.id}")
                    spec = CarSpecification(car_id=car.id)
                    session.add(spec)
                    await session.commit()
                    # Refresh to get the new ID
                    await session.refresh(spec)
                
                # Extract vehicle_type from description
                vehicle_type = None
                if car.description:
                    match = re.search(r'is a (.*?) car\.', car.description)
                    if match:
                        vehicle_type = match.group(1)
                
                # Extract HP and cylinders from engine string
                engine_hp = None
                engine_cylinders = None
                if spec.engine:
                    hp_match = re.search(r'(\d+)\s*HP', spec.engine)
                    if hp_match:
                        engine_hp = int(hp_match.group(1))
                    
                    cyl_match = re.search(r'(\d+)\s*cylinders', spec.engine)
                    if cyl_match:
                        engine_cylinders = int(cyl_match.group(1))
                
                # Prepare update dictionary with all fields
                update_dict = {
                    "engine_hp": engine_hp or 0,
                    "engine_cylinders": engine_cylinders or 0,
                    "engine_fuel_type": spec.fuel_type or "Unknown",
                    "transmission_type": spec.transmission or "Unknown",
                    "driven_wheels": "FWD",  # Default
                    "number_of_doors": 4,    # Default
                    "market_category": "Unknown",
                    "vehicle_size": "Midsize", # Default
                    "vehicle_style": spec.body_type or vehicle_type or "Unknown",
                    "highway_mpg": float(spec.mileage) if spec.mileage else 25.0, # Default
                    "city_mpg": 20,  # Default
                    "popularity": 100,  # Default
                    "msrp": car.price
                }
                
                # Update the specification
                await session.execute(
                    update(CarSpecification)
                    .where(CarSpecification.id == spec.id)
                    .values(**update_dict)
                )
                
                logger.info(f"Updated specification for car ID {car.id}")
            
            # Commit all changes
            await session.commit()
            logger.info("Successfully updated all car specifications")
            
        except Exception as e:
            logger.error(f"Error updating car specifications: {e}")
            await session.rollback()
            raise

async def main():
    """Main entry point"""
    try:
        await populate_car_specifications()
        logger.info("Car specification population completed successfully")
    except Exception as e:
        logger.error(f"Failed to populate car specifications: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 