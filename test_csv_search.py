import asyncio
import sys
import os

# Add the project root to the path
sys.path.append(os.getcwd())

# Try importing with debug output
print("Attempting to import car_service...")
try:
    from app.services.car_service import car_service
    print("Successfully imported car_service")
except Exception as e:
    print(f"Error importing car_service: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

async def test_csv_search():
    """Test searching for coupes in the CSV"""
    print("=== Testing CSV Search ===")
    
    try:
        # Search for coupes
        print("Searching for coupes...")
        results = await car_service.search_cars_from_csv(
            vehicle_style="Coupe",
            page=1,
            page_size=10,
            sort_by="Popularity",
            sort_direction="desc"
        )
        
        print(f"Found {results['total']} coupes")
        
        if results['total'] > 0:
            print("\nTop coupes:")
            for i, car in enumerate(results['items']):
                print(f"{i+1}. {car.get('brand', '')} {car.get('model', '')} - Style: {car.get('vehicle_style', '')}")
        else:
            print("No coupes found!")
        
        # Now search for one of the models mentioned in the incorrect response
        mentioned_models = ["Mazda 3", "FIAT 124 Spider", "Audi A4"]
        
        for model in mentioned_models:
            make, model_name = model.split(" ", 1)
            
            print(f"\n=== Testing search for {model} ===")
            results = await car_service.search_cars_from_csv(
                make=make,
                model=model_name,
                page=1,
                page_size=5
            )
            
            print(f"Found {results['total']} cars")
            
            if results['total'] > 0:
                print(f"\nCars found for {model}:")
                for i, car in enumerate(results['items']):
                    print(f"{i+1}. {car.get('brand', '')} {car.get('model', '')} - Style: {car.get('vehicle_style', '')}")
            else:
                print(f"No {model} found!")
    except Exception as e:
        print(f"Error in test_csv_search: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting test...")
    asyncio.run(test_csv_search())
    print("Test completed.") 