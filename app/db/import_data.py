import csv
import asyncio
from pathlib import Path
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from app.db.mysql import mysql, connect_to_mysql, close_mysql_connection
from app.core.logging import logger

async def import_car_data():
    try:
        # Connect to MySQL
        await connect_to_mysql()
        
        # Read CSV file
        csv_path = Path("data/cars_data.csv")
        if not csv_path.exists():
            logger.error(f"CSV file not found at {csv_path}")
            raise FileNotFoundError(f"CSV file not found at {csv_path}")

        with open(csv_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    # Insert into cars table
                    car_id = await mysql.execute_query(
                        """
                        INSERT INTO cars (
                            name, brand, model, year, price,
                            car_condition, car_type, description
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            f"{row['Make']} {row['Model']}",
                            row["Make"],
                            row["Model"],
                            int(row["Year"]),
                            float(row["MSRP"]),
                            "new",
                            "buy",
                            f"The {row['Make']} {row['Model']} is a {row.get('Vehicle Style', '')} car."
                        )
                    )

                    # Get the last inserted ID
                    result = await mysql.execute_query("SELECT LAST_INSERT_ID() as id")
                    car_id = result[0]['id']

                    # Insert into car_specifications table
                    await mysql.execute_query(
                        """
                        INSERT INTO car_specifications (
                            car_id, engine, transmission, fuel_type,
                            mileage, seating_capacity, body_type
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            car_id,
                            f"{row.get('Engine HP', '')} HP, {row.get('Engine Cylinders', '')} cylinders",
                            row.get("Transmission Type", ""),
                            row.get("Engine Fuel Type", ""),
                            float(row.get("highway MPG", 0)),
                            int(row.get("Number of Doors", 4)),
                            row.get("Vehicle Style", "")
                        )
                    )

                    # Insert into car_features table
                    features = []
                    if 'Market Category' in row and row['Market Category']:
                        features.extend([cat.strip() for cat in row['Market Category'].split(',') if cat.strip()])
                    
                    for feature in features:
                        await mysql.execute_query(
                            """
                            INSERT INTO car_features (car_id, feature_name)
                            VALUES (%s, %s)
                            """,
                            (car_id, feature)
                        )

                    logger.info(f"Successfully imported car: {row['Make']} {row['Model']}")

                except Exception as e:
                    logger.error(f"Error importing row: {row}. Error: {str(e)}")
                    continue

        logger.info("Data import completed successfully")

    except Exception as e:
        logger.error(f"Critical error during data import: {str(e)}")
        raise
    finally:
        # Close MySQL connection
        await close_mysql_connection()

async def main():
    try:
        await import_car_data()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Import process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")