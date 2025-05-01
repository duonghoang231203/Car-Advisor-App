import asyncio
from app.services.car_service import car_service

async def main():
    # Check filter options
    filter_options = await car_service.get_filter_options()
    print("Filter options:", filter_options)
    
    # Check if there are any vehicle styles in the car_specifications table
    async with car_service.session_factory() as session:
        from sqlalchemy import select, func
        from app.db.models import CarSpecification
        
        # Count total car specifications
        count_query = select(func.count()).select_from(CarSpecification)
        count_result = await session.execute(count_query)
        total_specs = count_result.scalar_one()
        print(f"Total car specifications in database: {total_specs}")
        
        # Get distinct body types
        from sqlalchemy import distinct
        body_types_query = select(distinct(CarSpecification.body_type))
        body_types_result = await session.execute(body_types_query)
        body_types = [bt for bt, in body_types_result.all() if bt]
        print(f"Distinct body types in database: {body_types}")

if __name__ == "__main__":
    asyncio.run(main())
