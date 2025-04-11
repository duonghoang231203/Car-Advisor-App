from fastapi import APIRouter, Depends, HTTPException, status, Query
from app.models.car import CarResponse, CarSearchParams
from app.services.car_service import CarService
from app.core.security import oauth2_scheme
from typing import List, Optional
from app.core.database import mysql
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()
car_service = CarService()

@router.get("/{car_id}", response_model=CarResponse)
async def get_car(
    car_id: str,
    token: Optional[str] = Depends(oauth2_scheme)
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

@router.get("/search", response_model=List[CarResponse])
async def search_cars(
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    brand: Optional[str] = None,
    type: Optional[str] = None,
    year: Optional[int] = None,
    condition: Optional[str] = None,
    rental: Optional[bool] = None,
    token: Optional[str] = Depends(oauth2_scheme)
) -> List[CarResponse]:
    """
    Search for cars based on criteria
    """
    # Create search parameters
    params = CarSearchParams(
        price_min=price_min,
        price_max=price_max,
        brand=brand,
        type=type,
        year=year,
        condition=condition,
        rental=rental
    )
    
    # Search cars
    cars = await car_service.search_cars(params)
    return cars

@router.post("/compare", response_model=List[CarResponse])
async def compare_cars(
    car_ids: List[str],
    token: Optional[str] = Depends(oauth2_scheme)
) -> List[CarResponse]:
    """
    Compare multiple cars
    """
    if len(car_ids) < 2 or len(car_ids) > 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You must provide 2 or 3 car IDs for comparison"
        )
    
    cars = await car_service.compare_cars(car_ids)
    
    if len(cars) != len(car_ids):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="One or more cars not found"
        )
    
    return cars

@router.get("/brands", response_model=List[str])
async def get_brands(
    token: Optional[str] = Depends(oauth2_scheme),
    session: AsyncSession = Depends(mysql.get_session)
) -> List[str]:
    """
    Get list of all car brands
    """
    # Get distinct brands from database
    result = await session.execute(select(Car.brand).distinct())
    brands = [brand for brand, in result.all()]
    return sorted(brands)

@router.get("/types", response_model=List[str])
async def get_types(
    token: Optional[str] = Depends(oauth2_scheme),
    session: AsyncSession = Depends(mysql.get_session)
) -> List[str]:
    """
    Get list of all car types (body types)
    """
    # Get distinct body types from database
    result = await session.execute(select(Car.specifications["body_type"].astext).distinct())
    types = [type_val for type_val, in result.all() if type_val]
    return sorted(types)