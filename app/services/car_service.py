from app.models.car import CarSearchParams, PaginatedCarResponse
from app.db.models import Car, CarSpecification
from typing import List, Optional
import pandas as pd
import os
import re
import random
from sqlalchemy import select, func, asc, desc, join
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from app.core.database import mysql

class CarService:
    def __init__(self):
        self.session_factory = mysql.session

    async def get_car_by_id(self, car_id: int) -> Optional[Car]:
        """Get a car by ID"""
        async with self.session_factory() as session:
            result = await session.execute(select(Car).where(Car.id == car_id))
            return result.scalars().first()

    async def _build_search_query(self, params: CarSearchParams):
        """Build a search query based on essential parameters"""
        # Use distinct on brand and model to avoid duplicates
        query = select(Car).distinct(Car.brand, Car.model)

        # Add filters based on parameters
        if params.brand:
            if params.partial_match:
                query = query.where(Car.brand.like(f"%{params.brand}%"))
            else:
                query = query.where(Car.brand == params.brand)

        # Add vehicle_style filtering if specified
        if params.vehicle_style:
            # Join with CarSpecification table to filter by body_type (vehicle_style)
            query = query.join(CarSpecification, Car.id == CarSpecification.car_id)

            if params.partial_match:
                query = query.where(CarSpecification.body_type.like(f"%{params.vehicle_style}%"))
            else:
                query = query.where(CarSpecification.body_type == params.vehicle_style)

        return query

    async def search_cars(self, params: CarSearchParams) -> List[Car]:
        """Search cars based on parameters (legacy method)"""
        async with self.session_factory() as session:
            query = await self._build_search_query(params)

            # Execute query
            result = await session.execute(query.limit(100))  # Limit to 100 results
            return result.scalars().all()

    async def search_cars_paginated(self, params: CarSearchParams) -> PaginatedCarResponse:
        """Search cars with pagination and sorting"""
        async with self.session_factory() as session:
            try:
                print(f"Search params: {params}")

                # Build the base query
                query = select(Car)
                print(f"Base query: {query}")

                # Apply brand filter if specified
                if params.brand:
                    if params.partial_match:
                        query = query.where(Car.brand.like(f"%{params.brand}%"))
                    else:
                        query = query.where(Car.brand == params.brand)
                    print(f"After brand filter: {query}")

                # Apply vehicle_style filter if specified
                if params.vehicle_style:
                    # Use the description field to filter by vehicle_style
                    # This is a workaround until we properly populate the car_specifications table
                    if params.partial_match:
                        query = query.where(Car.description.like(f"%{params.vehicle_style}%"))
                    else:
                        # For SUV, we need to be more flexible with the search
                        if params.vehicle_style.upper() == "SUV":
                            # Try to match any description that contains "SUV"
                            query = query.where(Car.description.like(f"%SUV%"))
                        else:
                            query = query.where(Car.description.like(f"%{params.vehicle_style}%"))
                    print(f"After vehicle_style filter: {query}")

                # Check if there are any cars in the database
                check_query = select(func.count()).select_from(Car)
                check_result = await session.execute(check_query)
                total_cars = check_result.scalar_one()
                print(f"Total cars in database: {total_cars}")

                # Create a subquery to count total results
                count_query = select(func.count()).select_from(query.subquery())
                total = await session.execute(count_query)
                total_count = total.scalar_one()
                print(f"Total count for query: {total_count}")

                # Apply sorting if specified
                if params.sort_by:
                    # Determine sort direction
                    sort_dir = desc if params.sort_direction and params.sort_direction.lower() == "desc" else asc

                    # Sort by model attribute
                    sort_attr = getattr(Car, params.sort_by, None)
                    if sort_attr:
                        query = query.order_by(sort_dir(sort_attr))
                else:
                    # Default sort by price
                    query = query.order_by(asc(Car.price))

                # Apply pagination
                offset = (params.page - 1) * params.page_size
                query = query.offset(offset).limit(params.page_size)
                print(f"Final query: {query}")

                # Execute query
                result = await session.execute(query)
                cars = result.scalars().all()
                print(f"Number of cars returned: {len(cars)}")

                # We want to show all cars that match the search criteria
                # No need to deduplicate for search results
                unique_car_list = cars
                print(f"Number of cars to display: {len(unique_car_list)}")

                # Calculate total pages
                total_pages = (total_count + params.page_size - 1) // params.page_size if total_count > 0 else 0

                # Convert SQLAlchemy models to dictionaries to avoid lazy loading issues
                car_dicts = []
                for car in unique_car_list:
                    # Extract vehicle_type from description
                    vehicle_type = None
                    if car.description:
                        # The description format is typically "The {brand} {model} is a {vehicle_style} car."
                        # Example: "The Ford Bronco II is a 2dr SUV car."
                        match = re.search(r'is a (.*?) car\.', car.description)
                        if match:
                            vehicle_type = match.group(1)

                    car_dict = {
                        "id": car.id,
                        "name": car.name,
                        "brand": car.brand,
                        "model": car.model,
                        "year": car.year,
                        "price": car.price,
                        "condition": car.condition,
                        "type": car.type,
                        "description": car.description,
                        "specifications": None,  # Set to None to avoid lazy loading issues
                        "image_urls": [],
                        "vehicle_type": vehicle_type  # Add the extracted vehicle_type
                    }
                    car_dicts.append(car_dict)

                # Create paginated response
                return PaginatedCarResponse(
                    items=car_dicts,
                    total=total_count,
                    page=params.page,
                    page_size=params.page_size,
                    total_pages=total_pages
                )
            except Exception as e:
                print(f"Error in search_cars_paginated: {e}")
                import traceback
                traceback.print_exc()
                # Return empty response in case of error
                return PaginatedCarResponse(
                    items=[],
                    total=0,
                    page=params.page,
                    page_size=params.page_size,
                    total_pages=0
                )

    async def compare_cars(self, car_ids: List[int]) -> List[dict]:
        """Get multiple cars for comparison"""
        async with self.session_factory() as session:
            # Get cars by IDs
            result = await session.execute(select(Car).where(Car.id.in_(car_ids)))
            cars = result.scalars().all()

            # Get specifications for these cars in a separate query
            car_ids_found = [car.id for car in cars]
            specs_result = await session.execute(
                select(CarSpecification).where(CarSpecification.car_id.in_(car_ids_found))
            )
            specs = specs_result.scalars().all()

            # Create a mapping of car_id to specification
            specs_by_car_id = {spec.car_id: spec for spec in specs}

            # Convert SQLAlchemy models to dictionaries to avoid lazy loading issues
            car_dicts = []
            for car in cars:
                # Extract vehicle_type from description
                vehicle_type = None
                if car.description:
                    # The description format is typically "The {brand} {model} is a {vehicle_style} car."
                    match = re.search(r'is a (.*?) car\.', car.description)
                    if match:
                        vehicle_type = match.group(1)

                # Get specification for this car
                spec = specs_by_car_id.get(car.id)
                spec_dict = None
                if spec:
                    spec_dict = {
                        "engine": spec.engine,
                        "transmission": spec.transmission,
                        "fuel_type": spec.fuel_type,
                        "mileage": float(spec.mileage) if spec.mileage else None,
                        "seating_capacity": spec.seating_capacity,
                        "body_type": spec.body_type
                    }

                car_dict = {
                    "id": car.id,
                    "name": car.name,
                    "brand": car.brand,
                    "model": car.model,
                    "year": car.year,
                    "price": car.price,
                    "condition": car.condition,
                    "type": car.type,
                    "description": car.description,
                    "specifications": spec_dict,
                    "image_urls": [],
                    "vehicle_type": vehicle_type
                }
                car_dicts.append(car_dict)

            return car_dicts

    async def get_filter_options(self) -> dict:
        """Get available options for essential filter fields"""
        async with self.session_factory() as session:
            # Get distinct brands
            brands_result = await session.execute(select(Car.brand).distinct())
            brands = [brand for brand, in brands_result.all() if brand]

            # Get distinct vehicle styles from car_specifications table
            vehicle_styles_result = await session.execute(select(CarSpecification.body_type).distinct())
            vehicle_styles = [style for style, in vehicle_styles_result.all() if style]

            # Return essential options
            return {
                "brands": sorted(brands),
                "vehicle_styles": sorted(vehicle_styles)
            }

    def _generate_random_price(self, make, vehicle_style):
        """Generate a random price based on car make and vehicle style"""
        # Base price ranges for different makes (brands)
        premium_brands = ["BMW", "Mercedes-Benz", "Audi", "Lexus", "Porsche", "Tesla", "Jaguar", "Land Rover"]
        mid_range_brands = ["Toyota", "Honda", "Volkswagen", "Mazda", "Subaru", "Hyundai", "Kia", "Nissan"]

        # Base price ranges for different vehicle styles
        luxury_styles = ["Luxury", "Convertible", "Coupe", "Exotic"]
        suv_styles = ["SUV", "Crossover", "Wagon"]

        # Set base price range based on make
        if make in premium_brands:
            base_min, base_max = 40000, 120000
        elif make in mid_range_brands:
            base_min, base_max = 20000, 45000
        else:
            base_min, base_max = 15000, 35000

        # Adjust price based on vehicle style
        if vehicle_style in luxury_styles:
            base_min *= 1.3
            base_max *= 1.5
        elif vehicle_style in suv_styles:
            base_min *= 1.1
            base_max *= 1.2

        # Generate random price within the range
        return round(random.uniform(base_min, base_max), 2)

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

                    # Extract market categories if available
                    market_categories = []
                    if 'Market Category' in row and row['Market Category']:
                        market_categories = [cat.strip() for cat in row['Market Category'].split(',') if cat.strip()]

                    # Get vehicle style
                    vehicle_style = row.get("Vehicle Style", "")

                    # Generate random price based on make and vehicle style
                    price = self._generate_random_price(row["Make"], vehicle_style)

                    # Create the car object with all the new fields
                    car = Car(
                        name=f"{row['Make']} {row['Model']}",
                        brand=row["Make"],
                        model=row["Model"],
                        year=int(row.get("Year", 2022)),
                        price=price,  # Use randomly generated price
                        condition="new",  # Default to new
                        type="buy",  # Default to buy
                        description=f"The {row['Make']} {row['Model']} is a {vehicle_style} car."
                    )

                    # Create the car specification object
                    car_spec = CarSpecification(
                        engine=f"{row.get('Engine HP', '')} HP, {row.get('Engine Cylinders', '')} cylinders",
                        transmission=row.get("Transmission Type", ""),
                        fuel_type=row.get("Engine Fuel Type", ""),
                        mileage=float(row.get("highway MPG", 0)) if row.get("highway MPG") else None,  # Use highway MPG as mileage
                        seating_capacity=5,  # Default value
                        body_type=vehicle_style  # This is the vehicle_style
                    )

                    # Link the specification to the car
                    car.specifications = car_spec

                    # Add features if available
                    features = []

                    # Add standard features if available in the CSV
                    for feature in ["ABS", "Airbags", "AC", "Power Steering", "Power Windows"]:
                        if feature in df.columns and row.get(feature) == 1:
                            features.append(feature)

                    # Add market categories as features
                    features.extend(market_categories)

                    # We'll handle features in a separate table in the future
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