import asyncio
from app.services.rag_service import rag_service
from app.services.car_service import CarService

async def test_coupe_search():
    """Test the searching capability for coupes"""
    try:
        # 1. Test if the query is correctly identified as car-related
        query = "What are some Coupes?"
        is_car_related = await rag_service._is_car_related_query(query)
        print(f"Is 'What are some Coupes?' car-related: {is_car_related}")
        
        # 2. Test filter creation from query
        filters = rag_service._create_filters_from_query(query)
        print(f"Filters created from 'What are some Coupes?': {filters}")
        
        # 3. Test searching for coupes in the database
        car_service = CarService()
        search_results = await car_service.search_cars_from_csv(vehicle_style="coupe", page=1, page_size=5)
        
        print(f"Found {search_results['total']} coupes in the database")
        print(f"Sample of first {len(search_results['items'])} coupes:")
        
        for i, car in enumerate(search_results['items']):
            print(f"{i+1}. {car['brand']} {car['model']} - {car.get('vehicle_style', 'Unknown')}")
        
        # 4. Test complete RAG query processing
        rag_response = await rag_service.process_query(query)
        print("\nRAG response to 'What are some Coupes?':")
        print(f"Response: {rag_response['response'][:100]}...")  # Show just the beginning
        print(f"Number of suggestions: {len(rag_response['suggestions'])}")
        
        for i, suggestion in enumerate(rag_response['suggestions']):
            print(f"Suggestion {i+1}: {suggestion['brand']} {suggestion['model']}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_coupe_search()) 