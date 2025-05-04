from fastapi import APIRouter, Depends, HTTPException, status, Query, Body, Path
from app.models.car import CarResponse, CarSearchParams, PaginatedCarResponse
from app.db.models import Car
from app.services.car_service import CarService
# Authentication removed
# from app.core.security import oauth2_scheme
from typing import List, Optional, Dict, Any
from app.core.database import mysql
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.logging import logger
from app.core.monitoring import timing_decorator

router = APIRouter()
car_service = CarService()

@router.get("/search", response_model=Dict[str, Any])
@timing_decorator
async def search_cars(
    # Essential search fields
    brand: Optional[str] = Query(None, title="Brand", description="Filter by car make/brand (e.g., BMW, Toyota)"),
    make: Optional[str] = Query(None, title="Make", description="Filter by car make/brand (e.g., BMW, Toyota)"),
    model: Optional[str] = Query(None, title="Model", description="Filter by car model (e.g., Corolla, 3 Series)"),
    year: Optional[int] = Query(None, title="Year", description="Filter by car year", gt=1900, lt=2100),
    min_price: Optional[float] = Query(None, title="Minimum Price", description="Filter by minimum price", ge=0),
    max_price: Optional[float] = Query(None, title="Maximum Price", description="Filter by maximum price", ge=0),
    vehicle_style: Optional[str] = Query(None, title="Vehicle Style", description="Filter by vehicle style (e.g., Sedan, SUV)"),

    # Search query parameter
    search_query: Optional[str] = Query(None, title="Search Query", description="General search query to match across multiple fields"),

    # Pagination and sorting
    page: int = Query(1, title="Page", description="Page number (starting from 1)", ge=1),
    page_size: int = Query(20, title="Page Size", description="Number of items per page", ge=1, le=100),
    sort_by: Optional[str] = Query(None, title="Sort By", description="Field to sort by (e.g., 'MSRP', 'Year')"),
    sort_direction: Optional[str] = Query("asc", title="Sort Direction", description="Sort direction ('asc' or 'desc')"),
    partial_match: bool = Query(False, title="Partial Match", description="Enable partial text matching for string fields")
) -> Dict[str, Any]:
    """
    Search for cars based on various criteria with data from CSV file

    - **brand/make**: Filter by car make/brand (e.g., BMW, Toyota)
    - **model**: Filter by car model (e.g., Corolla, 3 Series)
    - **year**: Filter by car year
    - **min_price/max_price**: Filter by price range
    - **vehicle_style**: Filter by vehicle style (e.g., Sedan, SUV)

    Pagination and sorting:
    - **sort_by**: Field to sort results by (e.g., 'MSRP', 'Year')
    - **sort_direction**: Sort direction (asc or desc)
    - **page/page_size**: Pagination parameters
    - **partial_match**: Enable partial text matching for string fields

    Returns:
        Dict[str, Any]: A dictionary containing:
            - items: List of car objects matching the search criteria
            - total: Total number of cars matching the search criteria
            - page: Current page number
            - page_size: Number of items per page
            - total_pages: Total number of pages
    """
    # Use make parameter if provided, otherwise use brand
    actual_make = make if make is not None else brand

    # Search cars with pagination using CSV data
    return await car_service.search_cars_from_csv(
        make=actual_make,
        model=model,
        year=year,
        min_price=min_price,
        max_price=max_price,
        vehicle_style=vehicle_style,
        search_query=search_query,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_direction=sort_direction,
        partial_match=partial_match
    )

@router.get("/brands", response_model=List[str])
async def get_brands(
    session: AsyncSession = Depends(mysql.get_session)
) -> List[str]:
    """
    Get list of all car brands
    """
    # Get distinct brands from database
    result = await session.execute(select(Car.brand).distinct())
    brands = [brand for brand, in result.all()]
    return sorted(brands)

@router.get("/vehicle-styles", response_model=List[str])
async def get_vehicle_styles() -> List[str]:
    """
    Get list of all vehicle styles
    """
    # Get vehicle styles from the car_specifications table
    filter_options = await car_service.get_filter_options()
    return filter_options.get('vehicle_styles', [])

@router.get("/filter-options", response_model=dict)
async def get_filter_options(
    # token: Optional[str] = Depends(oauth2_scheme)
) -> dict:
    """
    Get available options for filter fields

    Returns a dictionary with available options for the essential filter fields:
    - brands: List of car makes/brands
    - types: List of car types/body styles
    - price_range: Min and max price values
    """
    return await car_service.get_filter_options()

@router.get("/all", response_model=List[Dict[str, Any]])
async def get_all_cars():
    """
    Get a list of all cars from the CSV data source with complete specifications

    Returns:
        List[Dict[str, Any]]: A list of car objects with complete specifications
    """
    try:
        # Load CSV data
        df = car_service.load_csv_data()

        # Convert DataFrame to list of dictionaries
        cars = []
        for idx, row in df.iterrows():
            try:
                car_dict = car_service.convert_row_to_car_response(row)
                cars.append(car_dict)
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                continue

        return cars
    except Exception as e:
        logger.error(f"Error in get_all_cars: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading car data: {str(e)}"
        )

@router.get("/detail/{car_id}", response_model=CarResponse)
async def get_car_detail(
    car_id: int
) -> CarResponse:
    """
    Get comprehensive details for a specific car

    Returns all available car information including:
    - Make, Model, Year
    - Engine details (Fuel Type, HP, Cylinders)
    - Transmission Type
    - Driven Wheels
    - Number of Doors
    - Market Category
    - Vehicle Size and Style
    - MPG (highway and city)
    - Popularity
    - MSRP (Manufacturer's Suggested Retail Price)
    """
    car_dict = await car_service.get_car_by_id(car_id)
    if not car_dict:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Car not found"
        )
    # Convert dictionary to CarResponse model
    return CarResponse(**car_dict)

@router.post("/compare", response_model=List[CarResponse])
async def compare_cars(
    car_ids: List[int] = Body(...)
) -> List[CarResponse]:
    """
    Compare multiple cars
    """
    if len(car_ids) < 2 or len(car_ids) > 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You must provide 2 or 3 car IDs for comparison"
        )

    car_dicts = await car_service.compare_cars(car_ids)

    if len(car_dicts) != len(car_ids):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="One or more cars not found"
        )

    # Convert dictionaries to CarResponse objects
    return [CarResponse(**car_dict) for car_dict in car_dicts]

# This route must be at the end to avoid catching other routes
@router.get("/{car_id}", response_model=CarResponse)
async def get_car(
    car_id: int
) -> CarResponse:
    """
    Get details for a specific car with all specifications

    Returns detailed car information including:
    - Make, Model, Year
    - Engine details (Fuel Type, HP, Cylinders)
    - Transmission Type
    - Driven Wheels
    - Number of Doors
    - Market Category
    - Vehicle Size and Style
    - MPG (highway and city)
    - Popularity
    - MSRP
    """
    car_dict = await car_service.get_car_by_id(car_id)
    if not car_dict:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Car not found"
        )
    # Convert dictionary to CarResponse model
    return CarResponse(**car_dict)
