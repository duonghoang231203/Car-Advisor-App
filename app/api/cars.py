from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from app.models.car import CarResponse, CarSearchParams, PaginatedCarResponse
from app.db.models import Car
from app.services.car_service import CarService
# Authentication removed
# from app.core.security import oauth2_scheme
from typing import List, Optional
from app.core.database import mysql
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()
car_service = CarService()

@router.get("/search", response_model=PaginatedCarResponse)
async def search_cars(
    # Essential search fields
    brand: Optional[str] = None,
    vehicle_style: Optional[str] = None,

    # Pagination and sorting
    sort_by: Optional[str] = None,
    sort_direction: Optional[str] = "asc",
    page: int = 1,
    page_size: int = 20,
    partial_match: bool = False,
    # token: Optional[str] = Depends(oauth2_scheme)
) -> PaginatedCarResponse:
    """
    Search for cars based on essential criteria

    - **brand**: Filter by car make/brand (e.g., BMW, Toyota)
    - **vehicle_style**: Filter by vehicle style (e.g., Sedan, SUV)

    Pagination and sorting:
    - **sort_by**: Field to sort results by
    - **sort_direction**: Sort direction (asc or desc)
    - **page/page_size**: Pagination parameters
    - **partial_match**: Enable partial text matching for string fields
    """
    # Create search parameters
    params = CarSearchParams(
        brand=brand,
        vehicle_style=vehicle_style,
        sort_by=sort_by,
        sort_direction=sort_direction,
        page=page,
        page_size=page_size,
        partial_match=partial_match
    )

    # Search cars with pagination
    result = await car_service.search_cars_paginated(params)
    return result

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

@router.get("/{car_id}", response_model=CarResponse)
async def get_car(
    car_id: int
) -> CarResponse:
    """
    Get details for a specific car
    """
    car = await car_service.get_car_by_id(car_id)
    if not car:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Car not found"
        )
    return car

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
