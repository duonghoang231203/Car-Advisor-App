from typing import List, Dict, Any, Optional
import json
import asyncio
from openai import AsyncOpenAI
from app.config import settings
from app.services.car_service import CarService
from app.services.rag_service import RAGService
from app.services.web_search_service import web_search_service
from app.core.logging import logger
import re

class ChatService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
    async def chat(self, message: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Handle general chat conversation
        """
        try:
            messages = [
                {"role": "system", "content": "You are a friendly and knowledgeable car advisor assistant. Provide detailed, conversational responses that feel natural and engaging. When discussing cars, include specific details about features, performance, and what makes each option unique. Use a warm, helpful tone and structure your responses in a way that flows naturally, similar to how a car expert would explain things to a friend. Feel free to use line breaks and formatting to make the information easy to read."}
            ]
            
            if conversation_history:
                messages.extend(conversation_history)
                
            messages.append({"role": "user", "content": message})
            
            logger.info(f"Chat service processing message: {message}")
            
            async with asyncio.timeout(30):  # Set timeout to 30 seconds
                response = await self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )
            
            logger.info("Chat service response received successfully")
            
            return {
                "response": response.choices[0].message.content,
                "suggestions": [],
                "explanation": "General conversation response"
            }
        except asyncio.TimeoutError:
            logger.error("Chat request timed out")
            return {
                "response": "I apologize, but the request took too long to process. Please try again.",
                "suggestions": [],
                "explanation": "Request timeout"
            }
        except Exception as e:
            logger.error(f"Error in chat service: {str(e)}")
            return {
                "response": "I apologize, but I'm having trouble processing your request right now.",
                "suggestions": [],
                "explanation": "Error in chat processing"
            }

class FunctionCallingService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.car_service = CarService()
        self.rag_service = RAGService()
        self.chat_service = ChatService()
        
        # Define available functions
        self.available_functions = {
            "search_cars": {
                "name": "search_cars",
                "description": "Search for cars based on various criteria",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "make": {"type": "string", "description": "Car manufacturer"},
                        "model": {"type": "string", "description": "Car model"},
                        "year": {"type": "integer", "description": "Car year"},
                        "min_price": {"type": "number", "description": "Minimum price"},
                        "max_price": {"type": "number", "description": "Maximum price"},
                        "vehicle_style": {"type": "string", "description": "Type of vehicle (e.g., Sedan, SUV)"}
                    }
                }
            },
            "compare_cars": {
                "name": "compare_cars",
                "description": "Compare multiple cars by their IDs",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "car_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of car IDs to compare"
                        }
                    },
                    "required": ["car_ids"]
                }
            },
            "get_car_details": {
                "name": "get_car_details",
                "description": "Get detailed information about a specific car",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "car_id": {"type": "string", "description": "ID of the car"}
                    },
                    "required": ["car_id"]
                }
            },
            "search_car_info": {
                "name": "search_car_info",
                "description": "Search for general car information, news, and reviews from the web",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query about cars"},
                        "max_results": {"type": "integer", "description": "Maximum number of results to return", "default": 5}
                    },
                    "required": ["query"]
                }
            },
            "search_car_reviews": {
                "name": "search_car_reviews",
                "description": "Search for specific car reviews",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "brand": {"type": "string", "description": "Car brand"},
                        "model": {"type": "string", "description": "Car model"},
                        "max_results": {"type": "integer", "description": "Maximum number of results to return", "default": 3}
                    },
                    "required": ["brand", "model"]
                }
            },
            "search_car_comparison": {
                "name": "search_car_comparison",
                "description": "Search for comparison information between two cars",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "car1": {"type": "string", "description": "First car (brand model)"},
                        "car2": {"type": "string", "description": "Second car (brand model)"},
                        "max_results": {"type": "integer", "description": "Maximum number of results to return", "default": 3}
                    },
                    "required": ["car1", "car2"]
                }
            }
        }

    async def generate_natural_response(self, cars_data: List[Dict], query_type: str) -> str:
        """
        Generate a natural, conversational response using LLM
        """
        try:
            # Prepare the prompt for LLM
            if query_type == "search":
                if not cars_data:
                    prompt = """Create a friendly, helpful response for when no cars match the search criteria. 
                    The response should:
                    1. Acknowledge that no exact matches were found
                    2. Suggest alternative approaches (like broadening the search criteria)
                    3. Offer to help with a new search
                    4. Keep a positive and helpful tone
                    
                    Format the response naturally, as if a car expert is helping a friend."""
                else:
                    prompt = f"""Based on the following car data, create a natural, engaging response that highlights the key features and strengths of each car. 
                    Make it sound like a knowledgeable car expert talking to a friend. Include specific details about performance, features, and what makes each option unique.
                    End with an engaging question to encourage further discussion.

                    Car data:
                    {json.dumps(cars_data, indent=2)}

                    Format the response with proper line breaks and bullet points where appropriate."""
            elif query_type == "compare":
                if not cars_data:
                    prompt = """Create a friendly response for when there are no cars to compare. 
                    The response should:
                    1. Explain that we need at least two cars to make a comparison
                    2. Offer to help find cars to compare
                    3. Keep a helpful and encouraging tone
                    
                    Format the response naturally, as if a car expert is helping a friend."""
                else:
                    prompt = f"""Create a detailed comparison of these cars, highlighting their unique strengths and differences. 
                    Make it sound like a knowledgeable car expert providing insights to help someone make a decision.
                    End with a question about what aspects are most important to the user.

                    Car data:
                    {json.dumps(cars_data, indent=2)}

                    Format the response with proper line breaks and bullet points where appropriate."""
            else:  # details
                if not cars_data:
                    prompt = """Create a friendly response for when the requested car details cannot be found. 
                    The response should:
                    1. Acknowledge that the specific car couldn't be found
                    2. Offer to help search for similar cars
                    3. Keep a helpful and encouraging tone
                    
                    Format the response naturally, as if a car expert is helping a friend."""
                else:
                    prompt = f"""Create a detailed, engaging description of this car, highlighting its key features, performance, and unique characteristics.
                    Make it sound like a knowledgeable car expert sharing insights about a specific model.
                    End with a question about what aspects the user would like to know more about.

                    Car data:
                    {json.dumps(cars_data, indent=2)}

                    Format the response with proper line breaks and bullet points where appropriate."""

            # Call LLM to generate response
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a knowledgeable and enthusiastic car expert who provides detailed, engaging, and natural responses about cars."},
                    {"role": "user", "content": prompt}
                ]
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating natural response: {str(e)}")
            # Fallback to basic response if LLM fails
            if query_type == "search":
                return "I couldn't find any cars matching your criteria. Would you like to try a different search?"
            elif query_type == "compare":
                return "I couldn't find the cars to compare. Would you like to try searching for different cars?"
            else:
                return "I couldn't find the specific car you're looking for. Would you like to try a different search?"

    async def process_query(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Process a query using OpenAI function calling to determine the appropriate tool to use
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Prepare messages for the API call
            messages = [
                {"role": "system", "content": """You are a knowledgeable and enthusiastic car advisor assistant. Your goal is to provide detailed, engaging, and natural responses about cars. When discussing cars:

1. Start with a friendly introduction that acknowledges the user's interest
2. Provide detailed descriptions of each car, highlighting:
   - Key features and strengths
   - Performance characteristics
   - Design and style elements
   - Interior and comfort features
   - What makes each option unique
3. Use natural language and transitions between points
4. Include specific details about features, performance, and what makes each option unique
5. End with an engaging question to encourage further discussion
6. Use proper formatting with line breaks for readability
7. When mentioning cars, always include their make and model in a consistent format (e.g., "Honda Pilot", "Toyota Highlander")

Your responses should feel like a conversation with a car expert who's passionate about helping users find their perfect car."""}
            ]
            
            if conversation_history:
                messages.extend(conversation_history)
            messages.append({"role": "user", "content": query})

            logger.info("Calling OpenAI API with function calling...")
            async with asyncio.timeout(30):  # Set timeout to 30 seconds
                # Call OpenAI API with function calling
                response = await self.client.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    messages=messages,
                    tools=[{"type": "function", "function": func} for func in self.available_functions.values()],
                    tool_choice="auto"
                )

                response_message = response.choices[0].message
                logger.info(f"OpenAI API response received: {response_message}")

                # Check if the model wants to call a function
                if response_message.tool_calls:
                    tool_call = response_message.tool_calls[0]
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"Function call detected: {function_name}")
                    logger.info(f"Function arguments: {function_args}")

                    # Execute the appropriate function
                    if function_name == "search_cars":
                        logger.info("Executing search_cars function...")
                        
                        # Convert arguments to CarSearchParams
                        from app.models.car import CarSearchParams
                        
                        search_params = CarSearchParams(
                            brand=function_args.get("make"),
                            vehicle_style=function_args.get("vehicle_style"),
                            partial_match=True,
                            page=1,
                            page_size=20  # Increase page size to get more diverse options
                        )
                        
                        # Use search_cars_paginated instead of search_cars_from_csv
                        result = await self.car_service.search_cars_paginated(search_params)
                        logger.info(f"Search results: {result}")
                        
                        items = result.items
                        
                        # Create suggestions from all cars
                        suggestions = []
                        seen_names = set()  # Track unique car names
                        
                        for car in items:
                            car_name = f"{car.brand} {car.model}"
                            
                            # Only add if this car name hasn't been seen before
                            if car_name.lower() not in seen_names:
                                suggestions.append({
                                    "id": str(car.id),  # Use the actual database ID
                                    "name": f"{car.brand} {car.model} ({car.year})",
                                    "brand": car.brand,
                                    "model": car.model,
                                    "price": int(car.price) if car.price else 0,
                                    "db_id": car.id  # Database id for querying
                                })
                                seen_names.add(car_name.lower())
                        
                        # If we don't have enough suggestions, try to find more cars
                        if len(suggestions) < 5:  # Changed from 3 to 5 to ensure at least 5 suggestions
                            # Try to get additional related car suggestions
                            try:
                                # Get vehicle style from first car if available
                                original_vehicle_style = None
                                if car in items and (hasattr(car, 'vehicle_style') or hasattr(car, 'vehicle_type')):
                                    original_vehicle_style = getattr(car, 'vehicle_style', None) or getattr(car, 'vehicle_type', None)
                                    logger.info(f"Original vehicle style: {original_vehicle_style}")
                                    
                                # If not available from object, try to get it from car details
                                if not original_vehicle_style and len(suggestions) > 0 and "db_id" in suggestions[0]:
                                    try:
                                        first_car = await self.car_service.get_car_by_id(car_id=suggestions[0]["db_id"])
                                        if first_car:
                                            original_vehicle_style = first_car.get("vehicle_style") or first_car.get("vehicle_type")
                                            logger.info(f"Found vehicle style from DB: {original_vehicle_style}")
                                    except Exception as e:
                                        logger.error(f"Error getting vehicle style: {e}")
                                
                                # First try to find cars from the same brand AND same vehicle style
                                if car in items and car.brand and original_vehicle_style:
                                    brand_search_params = CarSearchParams(
                                        brand=car.brand,
                                        vehicle_style=original_vehicle_style,
                                        partial_match=True,
                                        page=1,
                                        page_size=15  # Increased from 10 to 15
                                    )
                                    
                                    brand_result = await self.car_service.search_cars_paginated(brand_search_params)
                                    
                                    # Add diverse cars from same brand and vehicle style
                                    for brand_car in brand_result.items:
                                        # Skip the original car
                                        if str(brand_car.id) == str(car.id):
                                            continue
                                            
                                        car_name = f"{brand_car.brand} {brand_car.model}"
                                        if car_name.lower() not in seen_names:
                                            suggestions.append({
                                                "id": str(brand_car.id),
                                                "name": f"{brand_car.brand} {brand_car.model} ({brand_car.year})",
                                                "brand": brand_car.brand,
                                                "model": brand_car.model,
                                                "price": int(brand_car.price) if brand_car.price else 0,
                                                "db_id": brand_car.id
                                            })
                                            seen_names.add(car_name.lower())
                                            
                                            # Stop when we have enough diverse suggestions
                                            if len(suggestions) >= 5:
                                                break
                                
                                # If still not enough, extract car types mentioned in the query or use original style
                                if len(suggestions) < 5:
                                    vehicle_types = ["SUV", "Sedan", "Truck", "Crossover", "Hatchback", "Coupe"]
                                    mentioned_type = None
                                    
                                    # First, prioritize vehicle type found in query
                                    for vtype in vehicle_types:
                                        if vtype.lower() in query.lower():
                                            mentioned_type = vtype
                                            break
                                    
                                    # If no specific type in query, use the vehicle style from original car
                                    if not mentioned_type and original_vehicle_style:
                                        mentioned_type = original_vehicle_style
                                        logger.info(f"Using original vehicle style: {mentioned_type}")
                                    
                                    # If still no type, default to popular types
                                    if not mentioned_type:
                                        mentioned_type = "SUV"  # Default to SUV as most popular
                                        
                                    logger.info(f"Searching for additional cars with vehicle style: {mentioned_type}")
                                    
                                    # Search for cars of this type
                                    additional_search_params = CarSearchParams(
                                        vehicle_style=mentioned_type,
                                        partial_match=True,
                                        page=1,
                                        page_size=15  # Increased from 10 to 15
                                    )
                                    
                                    add_result = await self.car_service.search_cars_paginated(additional_search_params)
                                    
                                    # Add diverse cars from search results
                                    for similar_car in add_result.items:
                                        car_name = f"{similar_car.brand} {similar_car.model}"
                                        if car_name.lower() not in seen_names:
                                            suggestions.append({
                                                "id": str(similar_car.id),
                                                "name": f"{similar_car.brand} {similar_car.model} ({similar_car.year})",
                                                "brand": similar_car.brand,
                                                "model": similar_car.model,
                                                "price": int(similar_car.price) if similar_car.price else 0,
                                                "db_id": similar_car.id
                                            })
                                            seen_names.add(car_name.lower())
                                            
                                            # Stop when we have enough diverse suggestions
                                            if len(suggestions) >= 5:
                                                break
                                
                                # If still not enough, just get popular cars OF THE SAME VEHICLE TYPE
                                if len(suggestions) < 5 and original_vehicle_style:
                                    popular_search_params = CarSearchParams(
                                        vehicle_style=original_vehicle_style,
                                        partial_match=True,
                                        page=1,
                                        page_size=15  # Increased from 10 to 15
                                    )
                                    
                                    popular_result = await self.car_service.search_cars_paginated(popular_search_params)
                                    
                                    # Add diverse popular cars of the same vehicle type
                                    for pop_car in popular_result.items:
                                        car_name = f"{pop_car.brand} {pop_car.model}"
                                        if car_name.lower() not in seen_names:
                                            suggestions.append({
                                                "id": str(pop_car.id),
                                                "name": f"{pop_car.brand} {pop_car.model} ({pop_car.year})",
                                                "brand": pop_car.brand,
                                                "model": pop_car.model,
                                                "price": int(pop_car.price) if pop_car.price else 0,
                                                "db_id": pop_car.id
                                            })
                                            seen_names.add(car_name.lower())
                                            
                                            # Stop when we have enough diverse suggestions
                                            if len(suggestions) >= 5:
                                                break
                            except Exception as e:
                                logger.error(f"Error getting additional suggestions: {e}")
                        
                        # Generate natural response using LLM
                        car_data_for_llm = [
                            {
                                "Make": car.brand,
                                "Model": car.model,
                                "Year": car.year,
                                "MSRP": car.price,
                                "Vehicle Style": car.vehicle_type or car.vehicle_style or "",
                                "description": car.description
                            }
                            for car in items
                        ]
                        
                        natural_response = await self.generate_natural_response(car_data_for_llm, "search")
                        
                        return {
                            "response": natural_response,
                            "suggestions": suggestions,
                            "explanation": "Results from car search"
                        }
                    elif function_name == "compare_cars":
                        logger.info("Executing compare_cars function...")
                        result = await self.car_service.compare_cars(**function_args)
                        logger.info(f"Comparison results: {result}")
                        
                        cars = result.get('cars', [])
                        
                        # Create suggestions from the comparison results
                        suggestions = []
                        seen_names = set()  # Track unique car names
                        
                        for car in cars:
                            car_name = f"{car['brand']} {car['model']}"
                            
                            # Only add if this car name hasn't been seen before
                            if car_name.lower() not in seen_names:
                                suggestions.append({
                                    "id": str(car['id']),  # Use the actual database ID
                                    "name": f"{car['brand']} {car['model']} ({car['year']})",
                                    "brand": car['brand'],
                                    "model": car['model'],
                                    "price": int(car['price']) if car['price'] else 0,
                                    "db_id": car['id']  # Database id for querying
                                })
                                seen_names.add(car_name.lower())
                        
                        # Generate natural response using LLM
                        car_data_for_llm = [
                            {
                                "Make": car['brand'],
                                "Model": car['model'],
                                "Year": car['year'],
                                "MSRP": car['price'],
                                "Vehicle Style": car.get('vehicle_type', '') or car.get('vehicle_style', ''),
                                "description": car.get('description', '')
                            }
                            for car in cars
                        ]
                        
                        natural_response = await self.generate_natural_response(car_data_for_llm, "compare")
                        
                        return {
                            "response": natural_response,
                            "suggestions": suggestions,
                            "explanation": "Car comparison results"
                        }
                    elif function_name == "get_car_details":
                        logger.info("Executing get_car_details function...")
                        result = await self.car_service.get_car_by_id(**function_args)
                        logger.info(f"Car details: {result}")
                        
                        if not result:
                            return {
                                "response": "I couldn't find the specific car you're looking for. Would you like to try a different search?",
                                "suggestions": [],
                                "explanation": "Car not found"
                            }
                        
                        # Create suggestion for the car
                        suggestion = {
                            "id": str(result['id']),  # Use the actual database ID
                            "name": f"{result['brand']} {result['model']} ({result['year']})",
                            "brand": result['brand'],
                            "model": result['model'],
                            "price": int(result['price']) if result['price'] else 0,
                            "db_id": result['id']  # Database id for querying
                        }
                        
                        # Convert to format expected by generate_natural_response
                        car_data_for_llm = [{
                            "Make": result['brand'],
                            "Model": result['model'],
                            "Year": result['year'],
                            "MSRP": result['price'],
                            "Vehicle Style": result.get('vehicle_type', '') or result.get('vehicle_style', ''),
                            "description": result.get('description', '')
                        }]
                        
                        # Generate natural response using LLM
                        natural_response = await self.generate_natural_response(car_data_for_llm, "details")
                        
                        return {
                            "response": natural_response,
                            "suggestions": [suggestion],
                            "explanation": "Car details"
                        }
                    elif function_name == "search_car_info":
                        logger.info("Executing search_car_info function...")
                        results = await web_search_service.search_car_info(**function_args)
                        logger.info(f"Web search results: {results}")
                        formatted_results = "\n".join([f"- {r['title']}\n  {r['snippet']}\n  {r['link']}" for r in results])
                        return {
                            "response": f"I've gathered some interesting information about your query:\n\n{formatted_results}\n\nThis should give you a good overview. Would you like me to dive deeper into any particular aspect?",
                            "suggestions": [],
                            "explanation": "Web search results for car information"
                        }
                    elif function_name == "search_car_reviews":
                        logger.info("Executing search_car_reviews function...")
                        results = await web_search_service.search_car_reviews(**function_args)
                        logger.info(f"Review search results: {results}")
                        formatted_results = "\n".join([f"- {r['title']}\n  {r['snippet']}\n  {r['link']}" for r in results])
                        return {
                            "response": f"I've found some insightful reviews for the {function_args['brand']} {function_args['model']}:\n\n{formatted_results}\n\nThese reviews should give you a good sense of what owners and experts think. Would you like to know more about any specific aspect?",
                            "suggestions": [],
                            "explanation": "Web search results for car reviews"
                        }
                    elif function_name == "search_car_comparison":
                        logger.info("Executing search_car_comparison function...")
                        results = await web_search_service.search_car_comparison(**function_args)
                        logger.info(f"Comparison search results: {results}")
                        formatted_results = "\n".join([f"- {r['title']}\n  {r['snippet']}\n  {r['link']}" for r in results])
                        return {
                            "response": f"Let me share what I found when comparing the {function_args['car1']} and {function_args['car2']}:\n\n{formatted_results}\n\nEach car has its own strengths. What aspects are most important to you in making your decision?",
                            "suggestions": [],
                            "explanation": "Web search results for car comparison"
                        }
                else:
                    logger.info("No function call detected, falling back to chat service")
                    chat_response = await self.chat_service.chat(query, conversation_history)
                    
                    # Extract car information from the response
                    car_pattern = r'\*\*([A-Za-z]+ [A-Za-z0-9-]+)\*\*'
                    cars = re.findall(car_pattern, chat_response["response"])
                    
                    # Create suggestions for each car mentioned
                    suggestions = []
                    seen_names = set()  # Track unique car names
                    
                    for car_name in cars:
                        # Skip if we've already seen this car name
                        if car_name.lower() in seen_names:
                            continue
                            
                        # Split car name into make and model
                        parts = car_name.split()
                        if len(parts) >= 2:
                            make = parts[0]
                            model = ' '.join(parts[1:])
                            
                            # Search for the car in database
                            try:
                                # Use search_cars_paginated instead of search_cars_from_csv
                                from app.models.car import CarSearchParams
                                
                                search_params = CarSearchParams(
                                    brand=make,
                                    partial_match=True,
                                    page=1,
                                    page_size=10  # Increase page size for more options
                                )
                                
                                search_result = await self.car_service.search_cars_paginated(search_params)
                                
                                # Filter for the specific model
                                matching_car = None
                                for car in search_result.items:
                                    if model.lower() in car.model.lower():
                                        matching_car = car
                                        break
                                
                                if matching_car:
                                    # Use the first matching car from database
                                    logger.info(f"Found car in database: {matching_car}")
                                    
                                    suggestions.append({
                                        "id": str(matching_car.id),  # Use the actual database ID
                                        "name": car_name,
                                        "brand": make,
                                        "model": model,
                                        "price": int(matching_car.price) if matching_car.price else 0,
                                        "db_id": matching_car.id  # Database id for querying
                                    })
                                    seen_names.add(car_name.lower())
                                    logger.info(f"Added suggestion with ID: {matching_car.id}")
                                else:
                                    # If not found in database, add with default values
                                    suggestions.append({
                                        "id": f"fallback_{len(suggestions) + 1}",  # Keep sequential ID as fallback
                                        "name": car_name,
                                        "brand": make,
                                        "model": model,
                                        "price": 0,
                                        "db_id": ""
                                    })
                                    seen_names.add(car_name.lower())
                                    logger.info(f"No database match found for {car_name}")
                            except Exception as e:
                                logger.error(f"Error getting details for {car_name}: {e}")
                                # Add with default values if error occurs
                                suggestions.append({
                                    "id": f"fallback_{len(suggestions) + 1}",  # Keep sequential ID as fallback
                                    "name": car_name,
                                    "brand": make,
                                    "model": model,
                                    "price": 0,
                                    "db_id": ""
                                })
                                seen_names.add(car_name.lower())
                    
                    # If we don't have enough suggestions, try to find more cars
                    if len(suggestions) < 5:  # Changed from 3 to 5 to ensure at least 5 suggestions
                        # Try to get additional related car suggestions
                        try:
                            # Get vehicle style from first car if available
                            original_vehicle_style = None
                            if car in items and (hasattr(car, 'vehicle_style') or hasattr(car, 'vehicle_type')):
                                original_vehicle_style = getattr(car, 'vehicle_style', None) or getattr(car, 'vehicle_type', None)
                                logger.info(f"Original vehicle style: {original_vehicle_style}")
                                
                            # If not available from object, try to get it from car details
                            if not original_vehicle_style and len(suggestions) > 0 and "db_id" in suggestions[0]:
                                try:
                                    first_car = await self.car_service.get_car_by_id(car_id=suggestions[0]["db_id"])
                                    if first_car:
                                        original_vehicle_style = first_car.get("vehicle_style") or first_car.get("vehicle_type")
                                        logger.info(f"Found vehicle style from DB: {original_vehicle_style}")
                                except Exception as e:
                                    logger.error(f"Error getting vehicle style: {e}")
                            
                            # First try to find cars from the same brand AND same vehicle style
                            if car in items and car.brand and original_vehicle_style:
                                brand_search_params = CarSearchParams(
                                    brand=car.brand,
                                    vehicle_style=original_vehicle_style,
                                    partial_match=True,
                                    page=1,
                                    page_size=15  # Increased from 10 to 15
                                )
                                
                                brand_result = await self.car_service.search_cars_paginated(brand_search_params)
                                
                                # Add diverse cars from same brand and vehicle style
                                for brand_car in brand_result.items:
                                    # Skip the original car
                                    if str(brand_car.id) == str(car.id):
                                        continue
                                        
                                    car_name = f"{brand_car.brand} {brand_car.model}"
                                    if car_name.lower() not in seen_names:
                                        suggestions.append({
                                            "id": str(brand_car.id),
                                            "name": f"{brand_car.brand} {brand_car.model} ({brand_car.year})",
                                            "brand": brand_car.brand,
                                            "model": brand_car.model,
                                            "price": int(brand_car.price) if brand_car.price else 0,
                                            "db_id": brand_car.id
                                        })
                                        seen_names.add(car_name.lower())
                                        
                                        # Stop when we have enough diverse suggestions
                                        if len(suggestions) >= 5:
                                            break
                            
                            # If still not enough, extract car types mentioned in the query or use original style
                            if len(suggestions) < 5:
                                vehicle_types = ["SUV", "Sedan", "Truck", "Crossover", "Hatchback", "Coupe"]
                                mentioned_type = None
                                
                                # First, prioritize vehicle type found in query
                                for vtype in vehicle_types:
                                    if vtype.lower() in query.lower():
                                        mentioned_type = vtype
                                        break
                                
                                # If no specific type in query, use the vehicle style from original car
                                if not mentioned_type and original_vehicle_style:
                                    mentioned_type = original_vehicle_style
                                    logger.info(f"Using original vehicle style: {mentioned_type}")
                                
                                # If still no type, default to popular types
                                if not mentioned_type:
                                    mentioned_type = "SUV"  # Default to SUV as most popular
                                    
                                logger.info(f"Searching for additional cars with vehicle style: {mentioned_type}")
                                
                                # Search for cars of this type
                                additional_search_params = CarSearchParams(
                                    vehicle_style=mentioned_type,
                                    partial_match=True,
                                    page=1,
                                    page_size=15  # Increased from 10 to 15
                                )
                                
                                add_result = await self.car_service.search_cars_paginated(additional_search_params)
                                
                                # Add diverse cars from search results
                                for similar_car in add_result.items:
                                    car_name = f"{similar_car.brand} {similar_car.model}"
                                    if car_name.lower() not in seen_names:
                                        suggestions.append({
                                            "id": str(similar_car.id),
                                            "name": f"{similar_car.brand} {similar_car.model} ({similar_car.year})",
                                            "brand": similar_car.brand,
                                            "model": similar_car.model,
                                            "price": int(similar_car.price) if similar_car.price else 0,
                                            "db_id": similar_car.id
                                        })
                                        seen_names.add(car_name.lower())
                                        
                                        # Stop when we have enough diverse suggestions
                                        if len(suggestions) >= 5:
                                            break
                                
                                # If still not enough, just get popular cars OF THE SAME VEHICLE TYPE
                                if len(suggestions) < 5 and original_vehicle_style:
                                    popular_search_params = CarSearchParams(
                                        vehicle_style=original_vehicle_style,
                                        partial_match=True,
                                        page=1,
                                        page_size=15  # Increased from 10 to 15
                                    )
                                    
                                    popular_result = await self.car_service.search_cars_paginated(popular_search_params)
                                    
                                    # Add diverse popular cars of the same vehicle type
                                    for pop_car in popular_result.items:
                                        car_name = f"{pop_car.brand} {pop_car.model}"
                                        if car_name.lower() not in seen_names:
                                            suggestions.append({
                                                "id": str(pop_car.id),
                                                "name": f"{pop_car.brand} {pop_car.model} ({pop_car.year})",
                                                "brand": pop_car.brand,
                                                "model": pop_car.model,
                                                "price": int(pop_car.price) if pop_car.price else 0,
                                                "db_id": pop_car.id
                                            })
                                            seen_names.add(car_name.lower())
                                            
                                            # Stop when we have enough diverse suggestions
                                            if len(suggestions) >= 5:
                                                break
                        except Exception as e:
                            logger.error(f"Error getting additional suggestions: {e}")
                    
                    return {
                        "response": chat_response["response"],
                        "suggestions": suggestions,
                        "explanation": "General conversation response"
                    }

        except asyncio.TimeoutError:
            logger.error("Function calling request timed out")
            return {
                "response": "I apologize, but the request took too long to process. Please try again.",
                "suggestions": [],
                "explanation": "Request timeout"
            }
        except Exception as e:
            logger.error(f"Error in function calling: {str(e)}")
            # Fallback to chat service in case of errors
            return await self.chat_service.chat(query, conversation_history)

# Create singleton instances
chat_service = ChatService()
function_calling_service = FunctionCallingService() 