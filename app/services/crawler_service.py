import requests
from bs4 import BeautifulSoup
import pandas as pd
from app.db.mongodb import mongodb
from app.db.vector_store import vector_db
from app.models.car import Car
import asyncio
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CrawlerService:
    def __init__(self):
        self.collection = mongodb.db.cars
    
    async def crawl_oto_com_vn(self):
        """Crawl car data from Oto.com.vn"""
        try:
            # Example URL - you'll need to adjust this based on the actual website structure
            url = "https://oto.com.vn/mua-ban-xe"
            
            # Send HTTP request
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract car listings
            car_listings = soup.find_all('div', class_='item-car')  # Adjust selector based on actual HTML
            
            cars = []
            for listing in car_listings:
                try:
                    # Extract car details - adjust selectors based on actual HTML
                    name = listing.find('h3', class_='title').text.strip()
                    price_text = listing.find('span', class_='price').text.strip()
                    price = float(price_text.replace('₫', '').replace(',', '').strip()) if price_text else 0
                    
                    # Parse brand and model from name
                    parts = name.split(' ', 1)
                    brand = parts[0] if parts else ""
                    model = parts[1] if len(parts) > 1 else ""
                    
                    # Extract image URL
                    img_tag = listing.find('img')
                    image_url = img_tag['src'] if img_tag and 'src' in img_tag.attrs else ""
                    
                    # Create car object
                    car = {
                        "name": name,
                        "brand": brand,
                        "model": model,
                        "year": 2022,  # Default, you might extract this from the listing
                        "price": price,
                        "condition": "new",  # Default, you might extract this from the listing
                        "type": "buy",  # Default, you might extract this from the listing
                        "specifications": {
                            "body_type": "",  # Extract from listing if available
                            "features": []
                        },
                        "description": "",  # Extract from listing if available
                        "image_urls": [image_url] if image_url else [],
                        "created_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                    
                    cars.append(car)
                
                except Exception as e:
                    logger.error(f"Error processing car listing: {e}")
                    continue
            
            # Insert or update cars in the database
            if cars:
                for car in cars:
                    # Check if car already exists (by name)
                    existing_car = await self.collection.find_one({"name": car["name"]})
                    
                    if existing_car:
                        # Update existing car
                        car["updated_at"] = datetime.utcnow()
                        await self.collection.update_one(
                            {"_id": existing_car["_id"]},
                            {"$set": car}
                        )
                    else:
                        # Insert new car
                        await self.collection.insert_one(car)
                
                # Update vector database
                await self.update_vector_db()
                
                print(f"Crawled and processed {len(cars)} cars from Oto.com.vn")
        
        except Exception as e:
            logger.error(f"Error crawling Oto.com.vn: {e}")
    
    async def crawl_carmudi_vn(self):
        """Crawl car data from Carmudi.vn"""
        try:
            # Example URL - you'll need to adjust this based on the actual website structure
            url = "https://www.carmudi.vn/mua-ban-o-to"
            
            # Send HTTP request
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract car listings - adjust selector based on actual HTML
            car_listings = soup.find_all('div', class_='car-listing')
            
            cars = []
            for listing in car_listings:
                try:
                    # Extract car details - adjust selectors based on actual HTML
                    name = listing.find('h2', class_='car-title').text.strip()
                    price_text = listing.find('span', class_='car-price').text.strip()
                    price = float(price_text.replace('₫', '').replace(',', '').strip()) if price_text else 0
                    
                    # Parse brand and model from name
                    parts = name.split(' ', 1)
                    brand = parts[0] if parts else ""
                    model = parts[1] if len(parts) > 1 else ""
                    
                    # Extract image URL
                    img_tag = listing.find('img')
                    image_url = img_tag['src'] if img_tag and 'src' in img_tag.attrs else ""
                    
                    # Create car object
                    car = {
                        "name": name,
                        "brand": brand,
                        "model": model,
                        "year": 2022,  # Default, you might extract this from the listing
                        "price": price,
                        "condition": "new",  # Default, you might extract this from the listing
                        "type": "buy",  # Default, you might extract this from the listing
                        "specifications": {
                            "body_type": "",  # Extract from listing if available
                            "features": []
                        },
                        "description": "",  # Extract from listing if available
                        "image_urls": [image_url] if image_url else [],
                        "created_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                    
                    cars.append(car)
                
                except Exception as e:
                    logger.error(f"Error processing car listing: {e}")
                    continue
            
            # Insert or update cars in the database
            if cars:
                for car in cars:
                    # Check if car already exists (by name)
                    existing_car = await self.collection.find_one({"name": car["name"]})
                    
                    if existing_car:
                        # Update existing car
                        car["updated_at"] = datetime.utcnow()
                        await self.collection.update_one(
                            {"_id": existing_car["_id"]},
                            {"$set": car}
                        )
                    else:
                        # Insert new car
                        await self.collection.insert_one(car)
                
                # Update vector database
                await self.update_vector_db()
                
                print(f"Crawled and processed {len(cars)} cars from Carmudi.vn")
        
        except Exception as e:
            logger.error(f"Error crawling Carmudi.vn: {e}")
    
    async def update_vector_db(self):
        """Update vector database with car data"""
        try:
            # Get all cars from MongoDB
            cursor = self.collection.find({})
            cars = await cursor.to_list(length=1000)  # Limit to 1000 cars
            
            # Prepare texts and metadata for vector DB
            texts = []
            metadatas = []
            
            for car in cars:
                # Convert ObjectId to string
                car_id = str(car["_id"])
                car["_id"] = car_id
                
                # Create a text representation of the car
                text = json.dumps(car)
                
                # Create metadata
                metadata = {
                    "car_id": car_id,
                    "name": car["name"],
                    "brand": car["brand"],
                    "model": car["model"],
                    "price": car["price"],
                    "type": car["type"]  # buy or rent
                }
                
                texts.append(text)
                metadatas.append(metadata)
            
            # Add to vector DB
            if texts:
                await vector_db.add_texts(texts=texts, metadatas=metadatas)
                print(f"Updated vector database with {len(texts)} cars")
        
        except Exception as e:
            logger.error(f"Error updating vector database: {e}")
    
    async def schedule_crawling(self):
        """Schedule periodic crawling"""
        while True:
            try:
                # Crawl data from different sources
                await self.crawl_oto_com_vn()
                await self.crawl_carmudi_vn()
                
                # Wait for 24 hours before crawling again
                await asyncio.sleep(24 * 60 * 60)
            
            except Exception as e:
                logger.error(f"Error in scheduled crawling: {e}")
                # Wait for 1 hour before retrying
                await asyncio.sleep(60 * 60)

crawler_service = CrawlerService() 