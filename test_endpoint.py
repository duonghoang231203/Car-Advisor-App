import asyncio
import httpx

async def test_vehicle_styles_endpoint():
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.get("/api/cars/vehicle-styles")
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        else:
            print(f"Error: {response.text}")

if __name__ == "__main__":
    asyncio.run(test_vehicle_styles_endpoint())
