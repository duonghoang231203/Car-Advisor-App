from app.models.car import Car, CarSearchParams
from typing import List, Optional
import pandas as pd
import os
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import mysql

class CarService:
    def __init__(self):
        self.session_factory = mysql.session
    
    async def get_car_by_id(self, car_id: str) -> Optional[Car]:
        """Get a car by ID"""
        async with self.session_factory() as session:
            result = await session.execute(select(Car).where(Car.id == car_id))
            return result.scalars().first()
    
    async def search_cars(self, params: CarSearchParams) -> List[Car]:
        """Search cars based on parameters"""
        async with self.session_factory() as session:
            query = select(Car)
            
            # Add filters based on parameters
            if params.price_min is not None:
                query = query.where(Car.price >= params.price_min)
            
            if params.price_max is not None:
                query = query.where(Car.price <= params.price_max)
            
            if params.brand:
                query = query.where(Car.brand == params.brand)
            
            if params.type:
                query = query.where(Car.specifications["body_type"].astext == params.type)
            
            if params.year:
                query = query.where(Car.year == params.year)
            
            if params.condition:
                query = query.where(Car.condition == params.condition)
            
            if params.rental is not None:
                car_type = "rent" if params.rental else "buy"
                query = query.where(Car.type == car_type)
            
            # Execute query
            result = await session.execute(query.limit(100))  # Limit to 100 results
            return result.scalars().all()
    
    async def compare_cars(self, car_ids: List[str]) -> List[Car]:
        """Get multiple cars for comparison"""
        async with self.session_factory() as session:
            result = await session.execute(select(Car).where(Car.id.in_(car_ids)))
            return result.scalars().all()
    
    async def load_initial_data(self):
        """Load initial car data from CSV file"""
        async with self.session_factory() as session:
            # Check if table has data
            result = await session.execute(select(Car).limit(1))
            if result.scalars().first() is not None:
                print("Car data already loaded, skipping...")
                return
            
            # Load data from CSV
            csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "cars_data.csv")
            if not os.path.exists(csv_path):
                print(f"CSV file not found at {csv_path}")
                return
            
            df = pd.read_csv(csv_path)
            
            # Transform data to match our model
            cars = []
            for _, row in df.iterrows():
                try:
                    # Map CSV columns to our model fields
                    # This will depend on the actual structure of your CSV
                    car = Car(
                        name=f"{row['Make']} {row['Model']}",
                        brand=row["Make"],
                        model=row["Model"],
                        year=int(row.get("Year", 2022)),
                        price=float(row.get("Price", 0)),
                        condition="new",  # Default to new
                        type="buy",  # Default to buy
                        specifications={
                            "engine": row.get("Engine", ""),
                            "transmission": row.get("Transmission", ""),
                            "fuel_type": row.get("Fuel Type", ""),
                            "mileage": float(row.get("Mileage", 0)),
                            "seating_capacity": int(row.get("Seating Capacity", 5)),
                            "body_type": row.get("Body Type", ""),
                            "features": []
                        },
                        description=f"The {row['Make']} {row['Model']} is a {row.get('Body Type', '')} car.",
                        image_urls=[]
                    )
                    
                    # Add features if available
                    features = []
                    for feature in ["ABS", "Airbags", "AC", "Power Steering", "Power Windows"]:
                        if feature in df.columns and row.get(feature) == 1:
                            features.append(feature)
                    
                    car.specifications["features"] = features
                    cars.append(car)
                
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
            
            # Insert data into MySQL
            if cars:
                session.add_all(cars)
                await session.commit()
                print(f"Loaded {len(cars)} cars into the database")

car_service = CarService()