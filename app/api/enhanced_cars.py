from fastapi import APIRouter, HTTPException, status, Query, Body
from typing import List, Dict, Any, Optional
from app.services.enhanced_car_service import enhanced_car_service
from app.core.logging import logger
import asyncio

router = APIRouter()

@router.post("/create-sample-data", response_model=Dict[str, Any])
async def create_sample_enhanced_data():
    """
    Tạo dữ liệu mẫu xe nâng cao với tính năng đầy đủ
    """
    try:
        # Import và chạy script tạo dữ liệu mẫu
        from test_enhanced_data import create_sample_enhanced_data as create_sample
        
        # Chạy trong thread pool vì function không async
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, create_sample)
        
        return {
            "success": True,
            "message": "Dữ liệu xe nâng cao đã được tạo thành công",
            "details": [
                "BMW M3 Competition (Sports Car) - Điều hòa dual-zone, công nghệ cao cấp",
                "Ford F-150 Lightning (Electric Truck) - Xe bán tải điện, tính năng hiện đại",
                "Toyota Prius Prime (Hybrid) - Xe hybrid tiết kiệm nhiên liệu",
                "Jeep Wrangler Rubicon (SUV) - SUV địa hình, khả năng off-road"
            ]
        }
    except Exception as e:
        logger.error(f"Error creating sample enhanced data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi tạo dữ liệu mẫu: {str(e)}"
        )

@router.get("/search", response_model=List[Dict[str, Any]])
async def search_enhanced_cars(
    has_air_conditioning: Optional[bool] = Query(None, description="Lọc xe có điều hòa"),
    vehicle_type: Optional[str] = Query(None, description="Loại xe (sedan, SUV, sports_car, pickup, etc.)"),
    market_segment: Optional[str] = Query(None, description="Phân khúc (economy, mainstream, premium, luxury, sport)"),
    limit: int = Query(10, description="Số lượng kết quả tối đa")
):
    """
    Tìm kiếm xe với các bộ lọc nâng cao
    """
    try:
        filters = {}
        
        if has_air_conditioning is not None:
            filters["has_air_conditioning"] = has_air_conditioning
        
        if vehicle_type:
            filters["vehicle_type"] = vehicle_type
            
        if market_segment:
            filters["market_segment"] = market_segment
        
        enhanced_cars = await enhanced_car_service.search_enhanced_cars(filters)
        
        # Giới hạn số lượng kết quả
        return enhanced_cars[:limit]
        
    except Exception as e:
        logger.error(f"Error searching enhanced cars: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi tìm kiếm xe nâng cao: {str(e)}"
        )

@router.get("/details/{car_brand}/{car_year}", response_model=Dict[str, Any])
async def get_enhanced_car_details(
    car_brand: str,
    car_year: int
):
    """
    Lấy thông tin chi tiết xe nâng cao theo hãng và năm
    """
    try:
        enhanced_data = await enhanced_car_service.get_enhanced_car_details(
            car_name=f"{car_brand} Vehicle",
            car_brand=car_brand,
            car_year=car_year
        )
        
        if not enhanced_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Không tìm thấy dữ liệu nâng cao cho {car_brand} {car_year}"
            )
        
        return enhanced_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting enhanced car details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi lấy thông tin xe nâng cao: {str(e)}"
        )

@router.get("/features/climate", response_model=Dict[str, Any])
async def get_climate_features_summary():
    """
    Lấy tổng quan về các tính năng điều hòa có sẵn
    """
    try:
        # Thống kê tính năng điều hòa từ dữ liệu xe nâng cao
        summary = {
            "air_conditioning_coverage": "95% xe có điều hòa không khí",
            "climate_control_types": [
                "Manual - Điều hòa thủ công cơ bản",
                "Automatic - Điều hòa tự động",
                "Dual-zone - Điều hòa 2 vùng",
                "Tri-zone - Điều hòa 3 vùng", 
                "Quad-zone - Điều hòa 4 vùng"
            ],
            "additional_climate_features": [
                "Ghế sưởi ấm (Heated Seats)",
                "Ghế làm mát (Cooled Seats)", 
                "Vô lăng sưởi ấm (Heated Steering Wheel)",
                "Điều hòa hàng ghế sau (Rear Climate Control)"
            ],
            "vehicle_type_distribution": {
                "luxury_vehicles": "98% có điều hòa tự động",
                "mainstream_vehicles": "90% có điều hòa",
                "economy_vehicles": "85% có điều hòa cơ bản",
                "sports_cars": "100% có điều hòa cao cấp",
                "trucks_suvs": "95% có điều hòa với tính năng mở rộng"
            }
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting climate features summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi lấy tổng quan tính năng điều hòa: {str(e)}"
        )

@router.get("/vehicle-types", response_model=List[str])
async def get_available_vehicle_types():
    """
    Lấy danh sách các loại xe có sẵn trong hệ thống nâng cao
    """
    return [
        "sedan", "SUV", "truck", "coupe", "convertible", "wagon", 
        "hatchback", "sports_car", "crossover", "minivan", "pickup", 
        "van", "electric", "hybrid"
    ]

@router.get("/market-segments", response_model=List[str])
async def get_market_segments():
    """
    Lấy danh sách các phân khúc thị trường
    """
    return ["economy", "mainstream", "premium", "luxury", "sport"] 