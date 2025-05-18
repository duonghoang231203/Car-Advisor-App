"""
Script to add missing columns to the car_specifications table.
This script will add the new columns for detailed car specifications.
"""

import asyncio
import logging
from sqlalchemy import text, inspect
from app.core.database import mysql, engine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SQL to add each column individually
ADD_COLUMNS_SQL = [
    "ALTER TABLE car_specifications ADD COLUMN engine_hp INT;",
    "ALTER TABLE car_specifications ADD COLUMN engine_cylinders INT;",
    "ALTER TABLE car_specifications ADD COLUMN engine_fuel_type VARCHAR(50);",
    "ALTER TABLE car_specifications ADD COLUMN transmission_type VARCHAR(50);",
    "ALTER TABLE car_specifications ADD COLUMN driven_wheels VARCHAR(50);",
    "ALTER TABLE car_specifications ADD COLUMN number_of_doors INT;",
    "ALTER TABLE car_specifications ADD COLUMN market_category VARCHAR(100);",
    "ALTER TABLE car_specifications ADD COLUMN vehicle_size VARCHAR(50);",
    "ALTER TABLE car_specifications ADD COLUMN vehicle_style VARCHAR(50);",
    "ALTER TABLE car_specifications ADD COLUMN highway_mpg FLOAT;",
    "ALTER TABLE car_specifications ADD COLUMN city_mpg INT;",
    "ALTER TABLE car_specifications ADD COLUMN popularity INT;",
    "ALTER TABLE car_specifications ADD COLUMN msrp FLOAT;"
]

async def add_missing_columns():
    """Add missing columns to the car_specifications table"""
    logger.info("Starting to add missing columns to car_specifications table...")
    
    async for session in mysql.get_session():
        try:
            # Get existing columns
            existing_columns = await get_existing_columns()
            logger.info(f"Existing columns: {existing_columns}")
            
            # Add each column if it doesn't exist
            for sql in ADD_COLUMNS_SQL:
                column_name = sql.split("ADD COLUMN")[1].split()[0]
                if column_name not in existing_columns:
                    try:
                        logger.info(f"Adding column: {column_name}")
                        await session.execute(text(sql))
                        await session.commit()
                    except Exception as e:
                        logger.warning(f"Error adding column {column_name}: {e}")
                        await session.rollback()
                        # Continue with next column
                else:
                    logger.info(f"Column {column_name} already exists")
            
            logger.info("Successfully added missing columns to car_specifications table")
        except Exception as e:
            logger.error(f"Error adding columns: {e}")
            await session.rollback()
            raise

async def get_existing_columns():
    """Get list of existing columns in car_specifications table"""
    async with engine.connect() as conn:
        # Use SHOW COLUMNS to get existing columns
        result = await conn.execute(text("SHOW COLUMNS FROM car_specifications"))
        columns = [row[0] for row in result.fetchall()]
        return columns

async def main():
    """Main entry point"""
    try:
        await add_missing_columns()
        logger.info("Column addition completed successfully")
    except Exception as e:
        logger.error(f"Failed to add columns: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 