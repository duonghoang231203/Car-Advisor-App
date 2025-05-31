from app.models.car import CarSearchParams, PaginatedCarResponse
from app.db.models import Car, CarSpecification
from typing import List, Optional, Dict, Any
import pandas as pd
import os
import re
import random
import time
from sqlalchemy import select, func, asc, desc, join
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from app.core.database import mysql
from app.core.logging import logger
from fastapi import HTTPException, status

class CarService:
    def __init__(self):
        self.session_factory = mysql.session

        # Cache for CSV data
        self._csv_data_cache = {
            "data": None,
            "last_updated": 0,
            "cache_ttl": 300  # Cache time-to-live in seconds (5 minutes)
        }

    def get_csv_path(self):
        """Get the path to the CSV file"""
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "cars_data.csv")
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found at {csv_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="CSV data file not found"
            )
        return csv_path

    def load_csv_data(self):
        """Load data from CSV file with caching and monitoring"""
        from app.core.monitoring import record_cache_hit, record_cache_miss

        current_time = time.time()

        # Check if cache is valid
        if (self._csv_data_cache["data"] is not None and
            current_time - self._csv_data_cache["last_updated"] < self._csv_data_cache["cache_ttl"]):
            logger.debug("Using cached CSV data")
            record_cache_hit()
            return self._csv_data_cache["data"]

        # Cache is invalid or empty, reload from file
        record_cache_miss()
        csv_path = self.get_csv_path()
        try:
            logger.info(f"Loading CSV data from {csv_path}")
            start_time = time.time()

            # Use simple CSV reading without categorical data types
            # to avoid type conversion issues
            df = pd.read_csv(csv_path, low_memory=False)

            # Pre-compute some common operations
            if 'Make' in df.columns and 'Model' in df.columns:
                df['full_name'] = df['Make'] + ' ' + df['Model']

            load_time = time.time() - start_time
            logger.info(f"CSV data loaded in {load_time:.2f} seconds")

            # Update cache
            self._csv_data_cache["data"] = df
            self._csv_data_cache["last_updated"] = current_time

            return df
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error loading CSV data: {str(e)}"
            )

    def convert_row_to_car_response(self, row, car_id=None):
        """Convert a DataFrame row to a CarResponse object with improved validation"""
        try:
            # Validate required fields
            required_fields = ["Make", "Model", "Year"]
            for field in required_fields:
                if field not in row or pd.isna(row[field]):
                    logger.error(f"Missing required field: {field}")
                    raise ValueError(f"Missing required field: {field}")

            # Extract market categories if available
            market_category = row.get("Market Category", "")
            if pd.isna(market_category):
                market_category = "Luxury"  # Default value

            # Handle vehicle style
            vehicle_style = row.get("Vehicle Style", "")
            if pd.isna(vehicle_style):
                vehicle_style = "Sedan"  # Default value

            # Handle engine HP with validation
            engine_hp = 0  # Default value
            if "Engine HP" in row and pd.notna(row["Engine HP"]):
                try:
                    hp_value = int(row["Engine HP"])
                    # Validate reasonable range
                    if 0 <= hp_value <= 2000:  # Reasonable HP range
                        engine_hp = hp_value
                except (ValueError, TypeError):
                    pass

            # Handle engine cylinders with validation
            engine_cylinders = 0  # Default value
            if "Engine Cylinders" in row and pd.notna(row["Engine Cylinders"]):
                try:
                    cyl_value = int(row["Engine Cylinders"])
                    # Validate reasonable range
                    if 0 <= cyl_value <= 16:  # Reasonable cylinder range
                        engine_cylinders = cyl_value
                except (ValueError, TypeError):
                    pass

            # Handle number of doors with validation
            number_of_doors = 4  # Default value
            if "Number of Doors" in row and pd.notna(row["Number of Doors"]):
                try:
                    doors_value = int(row["Number of Doors"])
                    # Validate reasonable range
                    if 1 <= doors_value <= 6:  # Reasonable door range
                        number_of_doors = doors_value
                except (ValueError, TypeError):
                    pass

            # Handle highway MPG with validation
            highway_mpg = 0.0  # Default value
            if "highway MPG" in row and pd.notna(row["highway MPG"]):
                try:
                    mpg_value = float(row["highway MPG"])
                    # Validate reasonable range
                    if 0 <= mpg_value <= 150:  # Reasonable MPG range
                        highway_mpg = mpg_value
                except (ValueError, TypeError):
                    pass

            # Handle city MPG with validation
            city_mpg = 0  # Default value
            if "city mpg" in row and pd.notna(row["city mpg"]):
                try:
                    mpg_value = int(row["city mpg"])
                    # Validate reasonable range
                    if 0 <= mpg_value <= 150:  # Reasonable MPG range
                        city_mpg = mpg_value
                except (ValueError, TypeError):
                    pass

            # Handle popularity with validation
            popularity = 0  # Default value
            if "Popularity" in row and pd.notna(row["Popularity"]):
                try:
                    pop_value = int(row["Popularity"])
                    # Validate reasonable range
                    if pop_value >= 0:  # Popularity should be non-negative
                        popularity = pop_value
                except (ValueError, TypeError):
                    pass

            # Handle MSRP with validation
            msrp = 0.0  # Default value
            if "MSRP" in row and pd.notna(row["MSRP"]):
                try:
                    msrp_value = float(row["MSRP"])
                    # Validate reasonable range
                    if 0 <= msrp_value <= 10000000:  # Reasonable price range
                        msrp = msrp_value
                except (ValueError, TypeError):
                    pass

            # Handle engine fuel type with validation
            engine_fuel_type = "Unknown"  # Default value
            if "Engine Fuel Type" in row and pd.notna(row["Engine Fuel Type"]):
                fuel_type = str(row["Engine Fuel Type"]).strip()
                if fuel_type:
                    engine_fuel_type = fuel_type

            # Handle transmission type with validation
            transmission_type = "Unknown"  # Default value
            if "Transmission Type" in row and pd.notna(row["Transmission Type"]):
                trans_type = str(row["Transmission Type"]).strip()
                if trans_type:
                    transmission_type = trans_type

            # Handle driven wheels with validation
            driven_wheels = "Unknown"  # Default value
            if "Driven_Wheels" in row and pd.notna(row["Driven_Wheels"]):
                wheels = str(row["Driven_Wheels"]).strip()
                if wheels:
                    driven_wheels = wheels

            # Handle vehicle size with validation
            vehicle_size = "Unknown"  # Default value
            if "Vehicle Size" in row and pd.notna(row["Vehicle Size"]):
                size = str(row["Vehicle Size"]).strip()
                if size:
                    vehicle_size = size

            # Create engine string with validation
            engine_str = f"{engine_hp} HP, {engine_cylinders} cylinders"

            # Create specifications object
            specifications = {
                "engine": engine_str,
                "transmission": transmission_type,
                "fuel_type": engine_fuel_type,
                "mileage": highway_mpg,
                "seating_capacity": 5,
                "body_type": vehicle_style,
                # Add all the detailed fields to specifications as well
                "engine_hp": engine_hp,
                "engine_cylinders": engine_cylinders,
                "engine_fuel_type": engine_fuel_type,
                "transmission_type": transmission_type,
                "driven_wheels": driven_wheels,
                "number_of_doors": number_of_doors,
                "market_category": market_category,
                "vehicle_size": vehicle_size,
                "vehicle_style": vehicle_style,
                "highway_mpg": highway_mpg,
                "city_mpg": city_mpg,
                "popularity": popularity,
                "msrp": msrp
            }

            # Create car response dictionary with validated data
            car_dict = {
                "id": car_id if car_id is not None else row.name,  # Use row index as ID if not provided
                "name": f"{row['Make']} {row['Model']}",
                "brand": row["Make"],
                "model": row["Model"],
                "year": int(row["Year"]),
                "price": msrp,
                "condition": "new",  # Default value
                "type": "buy",  # Default value
                "image_url": None,  # No image URL in CSV
                "seats": 5,  # Default value

                # Detailed car information
                "make": row["Make"],
                "engine_fuel_type": engine_fuel_type,
                "engine_hp": engine_hp,
                "engine_cylinders": engine_cylinders,
                "transmission_type": transmission_type,
                "driven_wheels": driven_wheels,
                "number_of_doors": number_of_doors,
                "market_category": market_category,
                "vehicle_size": vehicle_size,
                "vehicle_style": vehicle_style,
                "highway_mpg": highway_mpg,
                "city_mpg": city_mpg,
                "popularity": popularity,
                "msrp": msrp,

                # Add engine and transmission for compatibility
                "engine": engine_str,
                "transmission": transmission_type,
                "fuel_type": engine_fuel_type,

                # Add features as a dictionary
                "features": {
                    "safety": "ABS, Airbags",
                    "comfort": "Air Conditioning",
                    "technology": "Bluetooth"
                },

                # Add pros and cons
                "pros": "Good value",
                "cons": "Basic features",

                # Add description
                "description": f"The {row['Make']} {row['Model']} is a {vehicle_style} car.",

                # Empty image URLs array
                "image_urls": [],

                # Add vehicle_type (same as vehicle_style)
                "vehicle_type": vehicle_style,

                # Add specifications object - ensure it's never null
                "specifications": specifications
            }

            return car_dict
        except Exception as e:
            # Log error and return a minimal valid car object
            logger.error(f"Error creating car dictionary: {str(e)}")

            # Ensure we have valid Make and Model values
            make = row.get("Make", "Unknown") if isinstance(row, pd.Series) else "Unknown"
            model = row.get("Model", "Unknown") if isinstance(row, pd.Series) else "Unknown"
            year = int(row.get("Year", 2000)) if isinstance(row, pd.Series) and pd.notna(row.get("Year")) else 2000

            # Create a default specifications object
            default_specifications = {
                "engine": "0 HP, 0 cylinders",
                "transmission": "Unknown",
                "fuel_type": "Unknown",
                "mileage": 0.0,
                "seating_capacity": 5,
                "body_type": "Unknown",
                # Add all the detailed fields to specifications as well
                "engine_hp": 0,
                "engine_cylinders": 0,
                "engine_fuel_type": "Unknown",
                "transmission_type": "Unknown",
                "driven_wheels": "Unknown",
                "number_of_doors": 4,
                "market_category": "Unknown",
                "vehicle_size": "Unknown",
                "vehicle_style": "Unknown",
                "highway_mpg": 0.0,
                "city_mpg": 0,
                "popularity": 0,
                "msrp": 0.0
            }

            # Return a minimal valid car object
            return {
                "id": car_id if car_id is not None else 0,
                "name": f"{make} {model}",
                "brand": make,
                "model": model,
                "year": year,
                "price": 0.0,
                "condition": "new",
                "type": "buy",
                "image_url": None,
                "seats": 5,
                "make": make,
                "engine_fuel_type": "Unknown",
                "engine_hp": 0,
                "engine_cylinders": 0,
                "transmission_type": "Unknown",
                "driven_wheels": "Unknown",
                "number_of_doors": 4,
                "market_category": "Unknown",
                "vehicle_size": "Unknown",
                "vehicle_style": "Unknown",
                "highway_mpg": 0.0,
                "city_mpg": 0,
                "popularity": 0,
                "msrp": 0.0,
                "engine": "0 HP, 0 cylinders",
                "transmission": "Unknown",
                "fuel_type": "Unknown",
                "features": {
                    "safety": "ABS, Airbags",
                    "comfort": "Air Conditioning",
                    "technology": "Bluetooth"
                },
                "pros": "Unknown",
                "cons": "Unknown",
                "description": f"The {make} {model} is a car.",
                "image_urls": [],
                "vehicle_type": "Unknown",
                "specifications": default_specifications
            }

    async def get_car_by_id(self, car_id) -> dict:
        """Get a car by ID with detailed information from CSV data"""
        try:
            # Convert car_id to int if it's a string
            if isinstance(car_id, str):
                try:
                    car_id = int(car_id)
                except ValueError:
                    logger.error(f"Invalid car ID format: {car_id}")
                    return None

            logger.info(f"Fetching car with ID {car_id}")

            # For all cars, fetch from database
            async with self.session_factory() as session:
                # Query the car
                query = select(Car).where(Car.id == car_id)
                result = await session.execute(query)
                car = result.scalars().first()

                if not car:
                    logger.warning(f"Car with ID {car_id} not found in database")
                    return None

                # Extract vehicle_type from description
                vehicle_type = None
                if car.description:
                    # The description format is typically "The {brand} {model} is a {vehicle_style} car."
                    match = re.search(r'is a (.*?) car\.', car.description)
                    if match:
                        vehicle_type = match.group(1)

                # Get the car specification
                spec_result = await session.execute(
                    select(CarSpecification).where(CarSpecification.car_id == car.id)
                )
                spec = spec_result.scalars().first()

                # Parse engine details if available
                engine_hp = None
                engine_cylinders = None
                if spec and spec.engine:
                    # Try to extract HP and cylinders from the engine string
                    # Format is typically "X HP, Y cylinders"
                    hp_match = re.search(r'(\d+)\s*HP', spec.engine)
                    if hp_match:
                        engine_hp = int(hp_match.group(1))

                    cyl_match = re.search(r'(\d+)\s*cylinders', spec.engine)
                    if cyl_match:
                        engine_cylinders = int(cyl_match.group(1))

                # Create response dictionary with all detailed fields
                car_dict = {
                    "id": car.id,  # Keep the original ID from the database for consistency
                    "name": car.name,
                    "brand": car.brand,
                    "model": car.model,
                    "year": car.year,
                    "price": car.price,
                    "condition": car.condition,
                    "type": car.type,
                    "description": car.description,
                    "image_urls": [],
                    "vehicle_type": vehicle_type,

                    # Map brand to make for consistency
                    "make": car.brand,

                    # Add detailed specification fields - directly access the new fields
                    "engine_fuel_type": spec.engine_fuel_type if spec else None,
                    "engine_hp": spec.engine_hp if spec else engine_hp,
                    "engine_cylinders": spec.engine_cylinders if spec else engine_cylinders,
                    "transmission_type": spec.transmission_type if spec else spec.transmission if spec else None,
                    "driven_wheels": spec.driven_wheels if spec else None,
                    "number_of_doors": spec.number_of_doors if spec else None,
                    "market_category": spec.market_category if spec else None,
                    "vehicle_size": spec.vehicle_size if spec else None,
                    "vehicle_style": spec.vehicle_style if spec else spec.body_type if spec else vehicle_type,
                    "highway_mpg": spec.highway_mpg if spec else float(spec.mileage) if spec and spec.mileage else None,
                    "city_mpg": spec.city_mpg if spec else None,
                    "popularity": spec.popularity if spec else None,
                    "msrp": spec.msrp if spec else car.price,  # Using price as MSRP if not available

                    # Include the full specification object for backward compatibility
                    "specifications": {
                        "engine": spec.engine if spec else None,
                        "transmission": spec.transmission if spec else None,
                        "fuel_type": spec.fuel_type if spec else None,
                        "mileage": float(spec.mileage) if spec and spec.mileage else None,
                        "seating_capacity": spec.seating_capacity if spec else None,
                        "body_type": spec.body_type if spec else None,
                        # Add all the detailed fields to specifications as well
                        "engine_hp": spec.engine_hp if spec else engine_hp,
                        "engine_cylinders": spec.engine_cylinders if spec else engine_cylinders,
                        "engine_fuel_type": spec.engine_fuel_type if spec else spec.fuel_type if spec else None,
                        "transmission_type": spec.transmission_type if spec else spec.transmission if spec else None,
                        "driven_wheels": spec.driven_wheels if spec else None,
                        "number_of_doors": spec.number_of_doors if spec else None,
                        "market_category": spec.market_category if spec else None,
                        "vehicle_size": spec.vehicle_size if spec else None,
                        "vehicle_style": spec.vehicle_style if spec else spec.body_type if spec else vehicle_type,
                        "highway_mpg": spec.highway_mpg if spec else float(spec.mileage) if spec and spec.mileage else None,
                        "city_mpg": spec.city_mpg if spec else None,
                        "popularity": spec.popularity if spec else None,
                        "msrp": spec.msrp if spec else car.price
                    } if spec else None
                }

                return car_dict

            # Fallback for CSV data is removed as we want to use consistent IDs from the database
            
        except Exception as e:
            logger.error(f"Error getting car by ID {car_id}: {e}")
            return None

    async def _build_search_query(self, params: CarSearchParams):
        """Build the search query based on the provided parameters"""
        query = select(Car).options(selectinload(Car.specifications))
        
        # Build filters based on search parameters
        filters = []
        
        # Basic search fields
        if params.brand:
            if params.partial_match:
                filters.append(Car.brand.ilike(f"%{params.brand}%"))
            else:
                filters.append(Car.brand == params.brand)
                
        if params.model:
            if params.partial_match:
                filters.append(Car.model.ilike(f"%{params.model}%"))
            else:
                filters.append(Car.model == params.model)
                
        if params.year:
            filters.append(Car.year == params.year)
            
        if params.min_price:
            filters.append(Car.price >= params.min_price)
            
        if params.max_price:
            filters.append(Car.price <= params.max_price)
            
        if params.vehicle_style:
            if params.partial_match:
                filters.append(Car.specifications.has(CarSpecification.vehicle_style.ilike(f"%{params.vehicle_style}%")))
            else:
                filters.append(Car.specifications.has(CarSpecification.vehicle_style == params.vehicle_style))
        
        # Car specifications
        if params.number_of_doors:
            filters.append(Car.specifications.has(CarSpecification.number_of_doors == params.number_of_doors))
            
        if params.engine_fuel_type:
            if params.partial_match:
                filters.append(Car.specifications.has(CarSpecification.engine_fuel_type.ilike(f"%{params.engine_fuel_type}%")))
            else:
                filters.append(Car.specifications.has(CarSpecification.engine_fuel_type == params.engine_fuel_type))
                
        if params.transmission_type:
            if params.partial_match:
                filters.append(Car.specifications.has(CarSpecification.transmission_type.ilike(f"%{params.transmission_type}%")))
            else:
                filters.append(Car.specifications.has(CarSpecification.transmission_type == params.transmission_type))
                
        if params.driven_wheels:
            if params.partial_match:
                filters.append(Car.specifications.has(CarSpecification.driven_wheels.ilike(f"%{params.driven_wheels}%")))
            else:
                filters.append(Car.specifications.has(CarSpecification.driven_wheels == params.driven_wheels))
                
        if params.market_category:
            if params.partial_match:
                filters.append(Car.specifications.has(CarSpecification.market_category.ilike(f"%{params.market_category}%")))
            else:
                filters.append(Car.specifications.has(CarSpecification.market_category == params.market_category))
                
        if params.vehicle_size:
            if params.partial_match:
                filters.append(Car.specifications.has(CarSpecification.vehicle_size.ilike(f"%{params.vehicle_size}%")))
            else:
                filters.append(Car.specifications.has(CarSpecification.vehicle_size == params.vehicle_size))
        
        # Apply all filters
        if filters:
            query = query.where(*filters)
            
        # Apply sorting
        if params.sort_by:
            sort_field = getattr(Car, params.sort_by, None)
            if sort_field:
                if params.sort_direction.lower() == "desc":
                    query = query.order_by(desc(sort_field))
                else:
                    query = query.order_by(asc(sort_field))
        
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

                # Build the base query with join to car_specifications
                query = select(Car).join(CarSpecification, Car.id == CarSpecification.car_id)
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
                    if params.partial_match:
                        query = query.where(CarSpecification.vehicle_style.like(f"%{params.vehicle_style}%"))
                    else:
                        query = query.where(CarSpecification.vehicle_style == params.vehicle_style)
                    print(f"After vehicle_style filter: {query}")

                # Apply number_of_doors filter if specified
                if params.number_of_doors:
                    query = query.where(CarSpecification.number_of_doors == params.number_of_doors)
                    print(f"After number_of_doors filter: {query}")

                # Apply other specification filters
                if params.engine_fuel_type:
                    if params.partial_match:
                        query = query.where(CarSpecification.engine_fuel_type.like(f"%{params.engine_fuel_type}%"))
                    else:
                        query = query.where(CarSpecification.engine_fuel_type == params.engine_fuel_type)

                if params.transmission_type:
                    if params.partial_match:
                        query = query.where(CarSpecification.transmission_type.like(f"%{params.transmission_type}%"))
                    else:
                        query = query.where(CarSpecification.transmission_type == params.transmission_type)

                if params.driven_wheels:
                    if params.partial_match:
                        query = query.where(CarSpecification.driven_wheels.like(f"%{params.driven_wheels}%"))
                    else:
                        query = query.where(CarSpecification.driven_wheels == params.driven_wheels)

                if params.market_category:
                    if params.partial_match:
                        query = query.where(CarSpecification.market_category.like(f"%{params.market_category}%"))
                    else:
                        query = query.where(CarSpecification.market_category == params.market_category)

                if params.vehicle_size:
                    if params.partial_match:
                        query = query.where(CarSpecification.vehicle_size.like(f"%{params.vehicle_size}%"))
                    else:
                        query = query.where(CarSpecification.vehicle_size == params.vehicle_size)

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

                # Convert SQLAlchemy models to dictionaries
                car_dicts = []
                for car in cars:
                    # Get specifications for this car
                    spec_result = await session.execute(
                        select(CarSpecification).where(CarSpecification.car_id == car.id)
                    )
                    spec = spec_result.scalars().first()

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
                        "specifications": {
                            "engine": spec.engine if spec else None,
                            "transmission": spec.transmission if spec else None,
                            "fuel_type": spec.fuel_type if spec else None,
                            "mileage": float(spec.mileage) if spec and spec.mileage else None,
                            "seating_capacity": spec.seating_capacity if spec else None,
                            "body_type": spec.body_type if spec else None,
                            "engine_hp": spec.engine_hp if spec else None,
                            "engine_cylinders": spec.engine_cylinders if spec else None,
                            "engine_fuel_type": spec.engine_fuel_type if spec else None,
                            "transmission_type": spec.transmission_type if spec else None,
                            "driven_wheels": spec.driven_wheels if spec else None,
                            "number_of_doors": spec.number_of_doors if spec else None,
                            "market_category": spec.market_category if spec else None,
                            "vehicle_size": spec.vehicle_size if spec else None,
                            "vehicle_style": spec.vehicle_style if spec else None,
                            "highway_mpg": spec.highway_mpg if spec else None,
                            "city_mpg": spec.city_mpg if spec else None,
                            "popularity": spec.popularity if spec else None,
                            "msrp": spec.msrp if spec else None
                        } if spec else None,
                        "image_urls": [],
                        "vehicle_type": spec.vehicle_style if spec else None
                    }
                    car_dicts.append(car_dict)

                # Create paginated response
                return PaginatedCarResponse(
                    items=car_dicts,
                    total=total_count,
                    page=params.page,
                    page_size=params.page_size,
                    total_pages=(total_count + params.page_size - 1) // params.page_size if total_count > 0 else 0
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

    async def compare_cars(self, car_ids: List[str]) -> dict:
        """Get multiple cars for comparison with detailed information"""
        car_dicts = []
        specifications_dict = {}

        # Get all cars from the database without special cases
        async with self.session_factory() as session:
            # Get cars by IDs
            result = await session.execute(select(Car).where(Car.id.in_([int(car_id) for car_id in car_ids])))
            cars = result.scalars().all()

            # Get specifications for these cars in a separate query
            car_ids_found = [car.id for car in cars]
            specs_result = await session.execute(
                select(CarSpecification).where(CarSpecification.car_id.in_(car_ids_found))
            )
            specs = specs_result.scalars().all()

            # Create a mapping of car_id to specification
            specs_by_car_id = {spec.car_id: spec for spec in specs}

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

                # Parse engine details if available
                engine_hp = None
                engine_cylinders = None
                if spec and spec.engine:
                    # Try to extract HP and cylinders from the engine string
                    # Format is typically "X HP, Y cylinders"
                    hp_match = re.search(r'(\d+)\s*HP', spec.engine)
                    if hp_match:
                        engine_hp = int(hp_match.group(1))

                    cyl_match = re.search(r'(\d+)\s*cylinders', spec.engine)
                    if cyl_match:
                        engine_cylinders = int(cyl_match.group(1))

                # Create specification dictionary for the car
                car_specs = {}
                if spec:
                    car_specs = {
                        "car_type": vehicle_type,
                        "year": car.year,
                        "condition": car.condition,
                        "seats": spec.seating_capacity,
                        "engine": spec.engine,
                        "transmission": spec.transmission,
                        "fuel_type": spec.fuel_type,
                        "fuel_consumption": float(spec.mileage) if spec.mileage else None,
                        "engine_hp": engine_hp,
                        "engine_cylinders": engine_cylinders,
                        "engine_fuel_type": spec.fuel_type,
                        "transmission_type": spec.transmission,
                        "driven_wheels": spec.driven_wheels if hasattr(spec, 'driven_wheels') else None,
                        "number_of_doors": spec.number_of_doors if hasattr(spec, 'number_of_doors') else None,
                        "market_category": spec.market_category if hasattr(spec, 'market_category') else None,
                        "vehicle_size": spec.vehicle_size if hasattr(spec, 'vehicle_size') else None,
                        "vehicle_style": spec.body_type if spec.body_type else vehicle_type,
                        "highway_mpg": float(spec.mileage) if spec.mileage else None,
                        "city_mpg": spec.city_mpg if hasattr(spec, 'city_mpg') else None,
                        "popularity": spec.popularity if hasattr(spec, 'popularity') else None,
                        "msrp": car.price
                    }
                else:
                    # Provide default values if no specifications are found
                    car_specs = {
                        "car_type": vehicle_type,
                        "year": car.year,
                        "condition": car.condition,
                        "seats": 5,
                        "engine": "Unknown",
                        "transmission": "Unknown",
                        "fuel_type": "Unknown",
                        "fuel_consumption": None
                    }

                # Add this car's specifications to the specifications dictionary
                specifications_dict[str(car.id)] = car_specs

                # Create response dictionary with all detailed fields
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
                    "image_urls": [],
                    "vehicle_type": vehicle_type,

                    # Map brand to make for consistency
                    "make": car.brand,

                    # Add detailed specification fields
                    "engine_fuel_type": spec.fuel_type if spec else None,
                    "engine_hp": engine_hp,
                    "engine_cylinders": engine_cylinders,
                    "transmission_type": spec.transmission if spec else None,
                    "driven_wheels": spec.driven_wheels if spec and hasattr(spec, 'driven_wheels') else None,
                    "number_of_doors": spec.number_of_doors if spec and hasattr(spec, 'number_of_doors') else None,
                    "market_category": spec.market_category if spec and hasattr(spec, 'market_category') else None,
                    "vehicle_size": spec.vehicle_size if spec and hasattr(spec, 'vehicle_size') else None,
                    "vehicle_style": spec.body_type if spec else vehicle_type,
                    "highway_mpg": spec.mileage if spec else None,  # Using mileage as highway_mpg
                    "city_mpg": spec.city_mpg if spec and hasattr(spec, 'city_mpg') else None,
                    "popularity": spec.popularity if spec and hasattr(spec, 'popularity') else None,
                    "msrp": car.price,  # Using price as MSRP if not available

                    # Include the full specification object for backward compatibility
                    "specifications": None  # Set to None as we'll use the new structure
                }
                car_dicts.append(car_dict)

        # Create the new response structure with a dedicated specifications section
        response = {
            "cars": car_dicts,
            "specifications": specifications_dict
        }

        return response

    async def search_cars_from_csv(self, make=None, model=None, year=None, min_price=None, max_price=None,
                                 vehicle_style=None, search_query=None, page=1, page_size=10,
                                 sort_by=None, sort_direction='asc', partial_match=False):
        """
        Search cars from CSV data with filters
        """
        try:
            # Load CSV data
            df = pd.read_csv('data/cars_data.csv')
            
            # Ensure we have an id column - use index if no id column exists
            if 'id' not in df.columns:
                df['id'] = df.index.astype(str)
            
            # Log unique vehicle styles for debugging
            logger.info(f"Available vehicle styles in database: {df['Vehicle Style'].unique().tolist()}")
            logger.info(f"Searching for vehicle style: {vehicle_style}")
            
            # Create a mask for filtering
            mask = pd.Series(True, index=df.index)
            
            # Apply filters
            if make:
                if partial_match:
                    mask &= df['Make'].str.lower().fillna('').str.contains(make.lower(), na=False)
                else:
                    mask &= df['Make'].str.lower().fillna('') == make.lower()
                    
            if model:
                if partial_match:
                    # For partial match, try different variations of the model name
                    model_variations = [
                        model.lower(),
                        model.lower().replace(' ', ''),
                        model.lower().replace(' ', '-'),
                        model.lower().replace('-', ' '),
                        model.lower().replace('series', '').strip(),
                        model.lower().replace('class', '').strip()
                    ]
                    model_mask = pd.Series(False, index=df.index)
                    for variation in model_variations:
                        model_mask |= df['Model'].str.lower().fillna('').str.contains(variation, na=False)
                    mask &= model_mask
                else:
                    # For exact match, try different variations
                    model_variations = [
                        model.lower(),
                        model.lower().replace(' ', ''),
                        model.lower().replace(' ', '-'),
                        model.lower().replace('-', ' '),
                        model.lower().replace('series', '').strip(),
                        model.lower().replace('class', '').strip()
                    ]
                    model_mask = pd.Series(False, index=df.index)
                    for variation in model_variations:
                        model_mask |= df['Model'].str.lower().fillna('') == variation
                    mask &= model_mask
                    
            if year:
                mask &= df['Year'].fillna(0) == year
                
            if min_price:
                mask &= df['MSRP'].fillna(0) >= min_price
                
            if max_price:
                mask &= df['MSRP'].fillna(0) <= max_price
                
            if vehicle_style:
                # Always use partial match for vehicle style
                vehicle_style = vehicle_style.lower()
                vehicle_style_variations = [
                    vehicle_style,
                    vehicle_style.replace(' ', ''),
                    vehicle_style.replace(' ', '-'),
                    vehicle_style.replace('-', ' '),
                    vehicle_style.replace('suv', 's.u.v'),
                    vehicle_style.replace('s.u.v', 'suv')
                ]
                vehicle_style_mask = pd.Series(False, index=df.index)
                for variation in vehicle_style_variations:
                    vehicle_style_mask |= df['Vehicle Style'].str.lower().fillna('').str.contains(variation, na=False)
                mask &= vehicle_style_mask
                logger.info(f"Found {vehicle_style_mask.sum()} matches for vehicle style")
                    
            if search_query:
                search_query = search_query.lower()
                search_mask = (
                    df['Make'].str.lower().fillna('').str.contains(search_query, na=False) |
                    df['Model'].str.lower().fillna('').str.contains(search_query, na=False) |
                    df['Vehicle Style'].str.lower().fillna('').str.contains(search_query, na=False)
                )
                mask &= search_mask
            
            # Apply the mask
            filtered_df = df[mask]
            
            # Log results for debugging
            logger.info(f"Total matches after filtering: {len(filtered_df)}")
            if len(filtered_df) > 0:
                logger.info(f"Sample matches: {filtered_df[['Make', 'Model', 'Vehicle Style', 'id']].head().to_dict('records')}")
            
            # Sort if specified
            if sort_by:
                if sort_by in df.columns:
                    filtered_df = filtered_df.sort_values(by=sort_by, ascending=(sort_direction == 'asc'))
            
            # Calculate pagination
            total = len(filtered_df)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            
            # Get paginated results
            paginated_df = filtered_df.iloc[start_idx:end_idx]
            
            # Convert to list of dictionaries with proper formatting
            items = []
            for _, row in paginated_df.iterrows():
                try:
                    # Create engine string
                    engine_hp = row['Engine HP'] if 'Engine HP' in row and pd.notna(row['Engine HP']) else 0
                    engine_cylinders = row['Engine Cylinders'] if 'Engine Cylinders' in row and pd.notna(row['Engine Cylinders']) else 0
                    engine_str = f"{engine_hp} HP, {engine_cylinders} cylinders"
                    
                    # Create specifications
                    specifications = {
                        "engine": engine_str,
                        "transmission": row['Transmission Type'] if 'Transmission Type' in row and pd.notna(row['Transmission Type']) else 'Unknown',
                        "fuel_type": row['Engine Fuel Type'] if 'Engine Fuel Type' in row and pd.notna(row['Engine Fuel Type']) else 'Unknown',
                        "mileage": float(row['highway MPG']) if 'highway MPG' in row and pd.notna(row['highway MPG']) else None,
                        "seating_capacity": 5,  # Default value
                        "body_type": row['Vehicle Style'] if 'Vehicle Style' in row and pd.notna(row['Vehicle Style']) else 'Unknown',
                        "engine_hp": int(engine_hp) if pd.notna(engine_hp) else None,
                        "engine_cylinders": int(engine_cylinders) if pd.notna(engine_cylinders) else None,
                        "engine_fuel_type": row['Engine Fuel Type'] if 'Engine Fuel Type' in row and pd.notna(row['Engine Fuel Type']) else 'Unknown',
                        "transmission_type": row['Transmission Type'] if 'Transmission Type' in row and pd.notna(row['Transmission Type']) else 'Unknown',
                        "driven_wheels": row['Driven_Wheels'] if 'Driven_Wheels' in row and pd.notna(row['Driven_Wheels']) else 'Unknown',
                        "number_of_doors": int(row['Number of Doors']) if 'Number of Doors' in row and pd.notna(row['Number of Doors']) else 4,
                        "market_category": row['Market Category'] if 'Market Category' in row and pd.notna(row['Market Category']) else 'Unknown',
                        "vehicle_size": row['Vehicle Size'] if 'Vehicle Size' in row and pd.notna(row['Vehicle Size']) else 'Unknown',
                        "vehicle_style": row['Vehicle Style'] if 'Vehicle Style' in row and pd.notna(row['Vehicle Style']) else 'Unknown',
                        "highway_mpg": float(row['highway MPG']) if 'highway MPG' in row and pd.notna(row['highway MPG']) else None,
                        "city_mpg": int(row['city mpg']) if 'city mpg' in row and pd.notna(row['city mpg']) else None,
                        "popularity": int(row['Popularity']) if 'Popularity' in row and pd.notna(row['Popularity']) else None,
                        "msrp": float(row['MSRP']) if 'MSRP' in row and pd.notna(row['MSRP']) else None
                    }
                    
                    # Create car item matching CarResponse model structure
                    car_item = {
                        "id": int(row['id']),  # Convert to int as required by model
                        "name": f"{row['Make']} {row['Model']}",
                        "brand": row['Make'],
                        "model": row['Model'],
                        "year": int(row['Year']) if pd.notna(row['Year']) else 2024,  # Default to current year if missing
                        "price": float(row['MSRP']) if pd.notna(row['MSRP']) else 0.0,  # Default to 0 if missing
                        "condition": "new",  # Default value
                        "type": "buy",  # Default value
                        "description": f"The {row['Make']} {row['Model']} is a {row['Vehicle Style'] if pd.notna(row['Vehicle Style']) else 'car'}.",
                        "image_urls": [],
                        "vehicle_type": row['Vehicle Style'] if pd.notna(row['Vehicle Style']) else 'Unknown',
                        "make": row['Make'],  # Same as brand
                        "engine_fuel_type": row['Engine Fuel Type'] if 'Engine Fuel Type' in row and pd.notna(row['Engine Fuel Type']) else 'Unknown',
                        "engine_hp": int(engine_hp) if pd.notna(engine_hp) else None,
                        "engine_cylinders": int(engine_cylinders) if pd.notna(engine_cylinders) else None,
                        "transmission_type": row['Transmission Type'] if 'Transmission Type' in row and pd.notna(row['Transmission Type']) else 'Unknown',
                        "driven_wheels": row['Driven_Wheels'] if 'Driven_Wheels' in row and pd.notna(row['Driven_Wheels']) else 'Unknown',
                        "number_of_doors": int(row['Number of Doors']) if 'Number of Doors' in row and pd.notna(row['Number of Doors']) else 4,
                        "market_category": row['Market Category'] if 'Market Category' in row and pd.notna(row['Market Category']) else 'Unknown',
                        "vehicle_size": row['Vehicle Size'] if 'Vehicle Size' in row and pd.notna(row['Vehicle Size']) else 'Unknown',
                        "vehicle_style": row['Vehicle Style'] if pd.notna(row['Vehicle Style']) else 'Unknown',
                        "highway_mpg": float(row['highway MPG']) if 'highway MPG' in row and pd.notna(row['highway MPG']) else None,
                        "city_mpg": int(row['city mpg']) if 'city mpg' in row and pd.notna(row['city mpg']) else None,
                        "popularity": int(row['Popularity']) if 'Popularity' in row and pd.notna(row['Popularity']) else None,
                        "msrp": float(row['MSRP']) if pd.notna(row['MSRP']) else None,
                        "specifications": specifications
                    }
                    items.append(car_item)
                except Exception as e:
                    logger.error(f"Error processing row: {e}")
                    continue
            
            return {
                "items": items,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size
            }
            
        except Exception as e:
            logger.error(f"Error in search_cars_from_csv: {e}")
            return {
                "items": [],
                "total": 0,
                "page": page,
                "page_size": page_size,
                "total_pages": 0
            }

    async def get_filter_options(self) -> dict:
        """Get available options for essential filter fields"""
        # First try to get options from CSV data
        try:
            df = self.load_csv_data()

            # Get distinct brands from CSV
            csv_brands = df["Make"].dropna().unique().tolist() if "Make" in df.columns else []

            # Get distinct vehicle styles from CSV
            csv_vehicle_styles = df["Vehicle Style"].dropna().unique().tolist() if "Vehicle Style" in df.columns else []

            # If we have data from CSV, return it
            if csv_brands and csv_vehicle_styles:
                return {
                    "brands": sorted(csv_brands),
                    "vehicle_styles": sorted(csv_vehicle_styles)
                }
        except Exception as e:
            logger.error(f"Error getting filter options from CSV: {e}")

        # Fallback to database if CSV data is not available
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

                    # Create the car specification object with all available fields
                    car_spec = CarSpecification(
                        engine=f"{row.get('Engine HP', '')} HP, {row.get('Engine Cylinders', '')} cylinders",
                        engine_hp=int(row.get('Engine HP', 0)) if row.get('Engine HP') and str(row.get('Engine HP')).isdigit() else None,
                        engine_cylinders=int(row.get('Engine Cylinders', 0)) if row.get('Engine Cylinders') and str(row.get('Engine Cylinders')).isdigit() else None,
                        engine_fuel_type=row.get("Engine Fuel Type", ""),
                        transmission=row.get("Transmission Type", ""),
                        transmission_type=row.get("Transmission Type", ""),
                        driven_wheels=row.get("Driven_Wheels", ""),
                        number_of_doors=int(row.get("Number of Doors", 4)) if row.get("Number of Doors") and str(row.get("Number of Doors")).isdigit() else 4,
                        market_category=row.get("Market Category", ""),
                        vehicle_size=row.get("Vehicle Size", ""),
                        vehicle_style=vehicle_style,
                        highway_mpg=float(row.get("highway MPG", 0)) if row.get("highway MPG") else None,
                        city_mpg=int(row.get("city mpg", 0)) if row.get("city mpg") and str(row.get("city mpg")).isdigit() else None,
                        popularity=int(row.get("Popularity", 0)) if row.get("Popularity") and str(row.get("Popularity")).isdigit() else None,
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