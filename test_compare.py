import asyncio
import httpx
import json

async def test_compare_endpoint():
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        # Define the request body - send a JSON object with car_ids key
        data = {"car_ids": [1, 2]}

        # Make the POST request
        response = await client.post("/api/cars/compare", json=data)

        # Print the response
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        else:
            print(f"Error: {response.text}")

if __name__ == "__main__":
    asyncio.run(test_compare_endpoint())
