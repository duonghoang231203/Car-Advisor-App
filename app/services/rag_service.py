from typing import Dict, List, Any
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableSequence
from app.db.vector_store import vector_db
from app.config import settings
from app.models.car import Car
from app.services.car_service import CarService
import json
from app.core.logging import logger
import re
import random
from pydantic import BaseModel, Field
from typing import Optional, List as TypedList
import asyncio

# Giữ tối thiểu import từ llama-index để tránh lỗi
from llama_index.core.schema import Document

class CarSuggestion(BaseModel):
    """Car suggestion model for recommendations"""
    id: str
    name: str
    brand: str
    model: str
    price: float
    image_url: Optional[str] = None
    reasons: TypedList[str] = Field(default_factory=list)

class RAGService:
    def __init__(self):
        """Initialize the RAG service"""
        self.initialize_llm()
        self.qa_chain = None
        self.initialize_qa_chain()
        self.car_service = CarService()

        # Khởi tạo adapter đơn giản
        self.vector_adapter = LlamaIndexVectorAdapter(vector_db)

        # Track previously suggested car IDs to avoid suggesting the same cars again
        self.previously_suggested_car_ids = set()

        logger.info("RAGService initialized with simplified vector adapter")

    def initialize_llm(self):
        """Initialize the language model"""
        try:
            if settings.LLM_TYPE == "openai":
                self.llm = ChatOpenAI(
                    model_name=settings.LLM_MODEL,
                    openai_api_key=settings.OPENAI_API_KEY,
                    temperature=0.7
                )

                # Create a more creative version of the LLM with higher temperature
                # for generating more varied responses
                self.creative_llm = ChatOpenAI(
                    model_name=settings.LLM_MODEL,
                    openai_api_key=settings.OPENAI_API_KEY,
                    temperature=0.9
                )
            else:
                # Fallback to OpenAI
                self.llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    openai_api_key=settings.OPENAI_API_KEY,
                    temperature=0.7
                )

                # Creative version with higher temperature
                self.creative_llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    openai_api_key=settings.OPENAI_API_KEY,
                    temperature=0.9
                )
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            # Fallback to a mock LLM for safety
            self.llm = None
            self.creative_llm = None

    def initialize_qa_chain(self):
        """Initialize the QA chain with multiple variations for more diverse responses"""
        try:
            # Standard template for initial questions
            standard_template = """
            You are a car advisor helping customers choose cars. Use the following context to answer the question.

            Context: {context}

            Conversation History:
            {conversation_history}

            Current Question: {query}

            Previously mentioned cars: {previously_mentioned_cars}

            IMPORTANT INSTRUCTIONS:
            1. ALWAYS discuss MULTIPLE car options (at least 3 different cars) in your initial responses.
            2. When answering initial questions, present and compare at least 3 different car options.
            3. If this is a follow-up question, you MUST refer to the same cars mentioned in previous messages.
            4. Do NOT introduce new cars unless specifically asked for alternatives.
            5. Maintain continuity with previous responses - if you recommended specific cars before, continue discussing those same cars.
            6. Be specific and reference details from the conversation history.
            7. If you're asked about features, specifications, or comparisons, refer to the exact cars mentioned earlier.
            8. If you don't know the answer, say that you don't know but still refer to the previously mentioned cars.
            9. NEVER recommend just one car - always provide multiple options for comparison.

            Remember to be conversational and natural in your response. Vary your response style to sound more human-like.
            """

            # Enthusiastic template for more engaging responses
            enthusiastic_template = """
            You are an enthusiastic car advisor who loves helping customers find their perfect vehicle.
            Use the following context to answer the question with energy and excitement.

            Context: {context}

            Conversation History:
            {conversation_history}

            Current Question: {query}

            Previously mentioned cars: {previously_mentioned_cars}

            IMPORTANT INSTRUCTIONS:
            1. ALWAYS discuss MULTIPLE car options (at least 3 different cars) in your initial responses.
            2. When answering initial questions, present and compare at least 3 different car options.
            3. If this is a follow-up question, you MUST refer to the same cars mentioned in previous messages.
            4. Do NOT introduce new cars unless specifically asked for alternatives.
            5. Maintain continuity with previous responses - if you recommended specific cars before, continue discussing those same cars.
            6. Be specific and reference details from the conversation history.
            7. If you're asked about features, specifications, or comparisons, refer to the exact cars mentioned earlier.
            8. NEVER recommend just one car - always provide multiple options for comparison.

            Provide an enthusiastic, helpful response based on the context and conversation history.
            Be conversational and engaging, showing genuine excitement about helping the user find the right car.
            If you don't know the answer, be honest but still refer to the previously mentioned cars.
            """

            # Concise template for direct, to-the-point responses
            concise_template = """
            You are a concise, efficient car advisor helping customers choose vehicles.
            Use the following context to answer the question directly and efficiently.

            Context: {context}

            Conversation History:
            {conversation_history}

            Current Question: {query}

            Previously mentioned cars: {previously_mentioned_cars}

            IMPORTANT INSTRUCTIONS:
            1. ALWAYS discuss MULTIPLE car options (at least 3 different cars) in your initial responses.
            2. When answering initial questions, present and compare at least 3 different car options.
            3. If this is a follow-up question, you MUST refer to the same cars mentioned in previous messages.
            4. Do NOT introduce new cars unless specifically asked for alternatives.
            5. Maintain continuity with previous responses - if you recommended specific cars before, continue discussing those same cars.
            6. Be specific and reference details from the conversation history.
            7. If you're asked about features, specifications, or comparisons, refer to the exact cars mentioned earlier.
            8. NEVER recommend just one car - always provide multiple options for comparison.

            Provide a clear, concise response that directly addresses the question.
            Focus on the most relevant information without unnecessary details.
            If you don't know the answer, say so briefly but still refer to the previously mentioned cars.
            """

            # Create prompts from templates
            standard_prompt = ChatPromptTemplate.from_template(standard_template)
            enthusiastic_prompt = ChatPromptTemplate.from_template(enthusiastic_template)
            concise_prompt = ChatPromptTemplate.from_template(concise_template)

            # Create the QA chains using RunnableSequence
            self.qa_chain = standard_prompt | self.llm
            self.enthusiastic_qa_chain = enthusiastic_prompt | self.creative_llm
            self.concise_qa_chain = concise_prompt | self.llm
        except Exception as e:
            logger.error(f"Error initializing QA chain: {e}")
            self.qa_chain = None
            self.enthusiastic_qa_chain = None
            self.concise_qa_chain = None

    async def _is_car_related_query(self, query):
        """
        Determine if the query is related to cars or just general conversation

        Args:
            query: The user query

        Returns:
            bool: True if the query is car-related, False otherwise
        """
        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower().strip()

        # List of common greetings and general phrases that are not car-related
        general_phrases = [
            "hello", "hi", "hey", "greetings", "good morning", "good afternoon",
            "good evening", "how are you", "what's up", "howdy", "nice to meet you",
            "how's it going", "whats up", "how are you doing", "thank you", "thanks",
            "bye", "goodbye", "see you", "talk to you later", "see you later", "have a good day"
        ]

        # Check if the query is just a simple greeting
        for phrase in general_phrases:
            if query_lower == phrase or query_lower.startswith(phrase + " "):
                return False

        # Check for car-related keywords
        car_keywords = [
            "car", "vehicle", "auto", "automobile", "drive", "driving", "suv",
            "sedan", "truck", "hatchback", "coupe", "convertible", "minivan",
            "wagon", "hybrid", "electric", "diesel", "gasoline", "fuel", "mpg",
            "horsepower", "engine", "transmission", "model", "brand", "manufacturer",
            "price", "cost", "buy", "purchase", "lease", "rent", "toyota", "honda",
            "ford", "chevrolet", "bmw", "mercedes", "audi", "lexus", "nissan",
            "hyundai", "kia", "subaru", "volkswagen", "mazda", "jeep", "tesla"
        ]

        for keyword in car_keywords:
            if keyword in query_lower:
                return True

        # Check if the query is asking about recommendations or comparisons
        recommendation_patterns = [
            "recommend", "suggestion", "advise", "compare", "versus", "vs",
            "better", "best", "which", "what car", "good for", "looking for"
        ]

        for pattern in recommendation_patterns:
            if pattern in query_lower:
                return True

        # Default to False for ambiguous queries
        return False

    async def process_query(self, query, conversation_history=None, k=10):
        """
        Process a user query and return a response with car suggestions

        Args:
            query: The current user query
            conversation_history: List of previous messages in the conversation
            k: Number of documents to retrieve
        """
        try:
            # Check if this is a general conversation query or a car-related query
            is_car_related = await self._is_car_related_query(query)

            # If this is a follow-up question to a previous car-related conversation,
            # still treat it as car-related even if it doesn't contain car keywords
            if not is_car_related and conversation_history and len(conversation_history) > 0:
                is_follow_up = await self._is_follow_up_question(query, conversation_history)
                if is_follow_up:
                    # Check if any previous assistant message was car-related (contained suggestions)
                    for msg in conversation_history:
                        if msg.role == "assistant" and self._extract_cars_from_text(msg.content):
                            is_car_related = True
                            break

            # If not car-related, return a simple response without car suggestions
            if not is_car_related:
                logger.info(f"Query '{query}' identified as general conversation, not car-related")

                # Format conversation history for the prompt
                formatted_history = ""
                if conversation_history and len(conversation_history) > 0:
                    for msg in conversation_history:
                        role = "User" if msg.role == "user" else "Assistant"
                        formatted_history += f"{role}: {msg.content}\n"

                # Generate a friendly general response
                general_prompt = f"""
                You are a helpful assistant in a car dealership.
                The user has sent a general message that is not about cars: "{query}"

                Conversation history:
                {formatted_history}

                Provide a friendly, brief response without suggesting any cars.
                Do not ask if they want car recommendations unless they've indicated interest in cars.
                """

                response_obj = await self.llm.ainvoke(general_prompt)
                response_text = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)

                # Return response without car suggestions
                return {
                    "response": response_text,
                    "suggestions": [],
                    "explanation": ""
                }

            # Format conversation history for the prompt
            formatted_history = ""
            previously_mentioned_cars = []

            if conversation_history and len(conversation_history) > 0:
                for msg in conversation_history:
                    role = "User" if msg.role == "user" else "Assistant"
                    formatted_history += f"{role}: {msg.content}\n"

                    # Extract car mentions from assistant messages to maintain consistency
                    if msg.role == "assistant":
                        # Look for car brand/model mentions in the assistant's responses
                        import re
                        # Common car brands to look for
                        car_brands = ["Toyota", "Honda", "Ford", "Chevrolet", "BMW", "Mercedes",
                                     "Audi", "Lexus", "Nissan", "Hyundai", "Kia", "Subaru",
                                     "Volkswagen", "Mazda", "Jeep", "Tesla", "Volvo"]

                        for brand in car_brands:
                            # Look for brand followed by model
                            matches = re.findall(f"{brand}\\s+[\\w-]+", msg.content)
                            for match in matches:
                                if match not in previously_mentioned_cars:
                                    previously_mentioned_cars.append(match)

            logger.info(f"Previously mentioned cars: {previously_mentioned_cars}")

            # Check if this is a follow-up question and if user is asking for more options
            is_follow_up, is_asking_for_more = await self._is_follow_up_question(query, conversation_history)
            logger.info(f"Query: '{query}' - Is follow-up: {is_follow_up} - Is asking for more: {is_asking_for_more}")

            # Enhance the query to make vector search more effective
            enhanced_query = await self._enhance_query(query, conversation_history)

            # If user is asking for more options, modify the query to encourage diversity
            if is_asking_for_more:
                # Add terms to encourage different results
                enhanced_query += " alternative different options"
                logger.info(f"User is asking for more options, modified query to: '{enhanced_query}'")

            logger.info(f"Enhanced query: '{enhanced_query}'")

            # Adjust search parameters based on whether this is a follow-up
            search_k = k
            if is_follow_up:
                # For follow-up questions, we might want to retrieve more documents
                # to ensure we have enough context to answer the follow-up
                search_k = k + 5

            # If asking for more options, increase search scope
            if is_asking_for_more:
                search_k = k + 10

            # Tạo bộ lọc nếu cần (ví dụ lọc theo loại xe được đề cập trong truy vấn)
            filters = self._create_filters_from_query(enhanced_query)

            # Step 1: Get relevant car documents using LlamaIndex adapter
            docs = await self.vector_adapter.query(enhanced_query, top_k=search_k, filters=filters)
            context = "\n".join([doc.page_content for doc in docs]) if docs else ""

            # Step 2: Generate explanation first
            explanation_prompt = f"""
            You are a car advisor helping customers choose between different car options.

            Based on the following information:

            User query: {query}

            Available context: {context}

            Please provide a detailed comparison of at least 3 specific car models (including brand and model names)
            that would be suitable for this user's needs. Your response should include specific car brands and models,
            their key features, and why they would be a good match for the user's needs.

            IMPORTANT: You must mention at least 3 SPECIFIC car models by their exact brand and model name (e.g., "Toyota Camry", "Honda Civic").
            """

            # Call the LLM to generate the explanation with specific car mentions
            if self.creative_llm:
                response_obj = await self.creative_llm.ainvoke(explanation_prompt)
                explanation = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
            else:
                response_obj = await self.llm.ainvoke(explanation_prompt)
                explanation = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)

            logger.info(f"Generated explanation with specific car mentions")

            # Step 3: Extract car mentions from the explanation
            extracted_cars = self._extract_cars_from_text(explanation)
            logger.info(f"Extracted cars from explanation: {extracted_cars}")

            # Step 4: Clean up extracted cars - remove invalid car models
            cleaned_cars = []
            invalid_models = ["safety", "sensing", "sense", "eyesight", "shield", "sync", "carplay", "android", "connect"]

            for car in extracted_cars:
                # Skip models that are actually safety features or technology packages
                if car['model'].lower() in invalid_models:
                    logger.info(f"Skipping non-car model: {car['name']}")
                    continue

                # Add to cleaned list
                cleaned_cars.append(car)

            # Filter out duplicates by name
            unique_car_names = set()
            filtered_cars = []

            for car in cleaned_cars:
                if car['name'].lower() not in unique_car_names:
                    unique_car_names.add(car['name'].lower())
                    filtered_cars.append(car)

            # Step 5: Look up these cars in the database and build suggestions
            suggestions = []
            used_names = set()  # Track used names to ensure uniqueness

            # Process each extracted car
            for car in filtered_cars:
                # Skip if we already have this exact car name
                if car['name'] in used_names:
                    continue

                # Get car data for this car
                car_data = await self._get_car_data_by_name(car['brand'], car['model'])

                if car_data and 'id' in car_data:
                    # Get car_id from the 'id' field
                    car_id = car_data.get('id')

                    # Double-check that this car exists in the database
                    car_db_data = await self._get_car_data(car_id)

                    if car_db_data:
                        # We have a confirmed database entry
                        logger.info(f"Found valid database entry for {car['name']}: ID {car_id}")

                        car_suggestion = CarSuggestion(
                            id=str(car_id),  # Convert ID to string for consistency
                            name=f"{car_db_data.get('brand', car['brand'])} {car_db_data.get('model', car['model'])}",
                            brand=car_db_data.get('brand', car['brand']),
                            model=car_db_data.get('model', car['model']),
                            price=float(car_db_data.get('msrp', car_data.get('MSRP', 0))),
                            image_url=car_db_data.get('image_url', None),
                            reasons=[f"Recommended {car_db_data.get('vehicle_style', 'vehicle')}"]
                        )

                        # Convert to dict using model_dump
                        suggestion_dict = car_suggestion.model_dump()

                        suggestions.append(suggestion_dict)
                        used_names.add(car['name'])

                        # Track this car ID as suggested
                        self.previously_suggested_car_ids.add(str(car_id))
                    else:
                        logger.warning(f"Car with ID {car_id} not found in database verification")
                else:
                    logger.warning(f"No valid car data found for {car['name']}")

            # If we don't have enough suggestions, create synthetic suggestions from extracted cars
            # Ensure we have at least 3 suggestions, but try to get up to 6 for better diversity
            if len(suggestions) < min(6, max(3, len(filtered_cars))):
                logger.info(f"Not enough suggestions from database ({len(suggestions)}), creating synthetic suggestions")

                # Get details about car type, price range, etc.
                vehicle_type = self._get_vehicle_type_from_query(query)

                # Create synthetic suggestions for cars not found in database
                for car in filtered_cars:
                    # Skip if we already have this car in suggestions
                    if car['name'] in used_names:
                        continue

                    # Generate reasonable price based on brand and vehicle type
                    price = self._estimate_car_price(car['brand'], car['model'], vehicle_type)

                    # Create a suggestion
                    car_suggestion = CarSuggestion(
                        id=f"synthetic_{len(suggestions)}",
                        name=car['name'],
                        brand=car['brand'],
                        model=car['model'],
                        price=price,
                        image_url=None,
                        reasons=[f"Recommended {vehicle_type}"]
                    )

                    # Convert to dict
                    suggestion_dict = car_suggestion.model_dump()

                    suggestions.append(suggestion_dict)
                    used_names.add(car['name'])

                    # Stop when we have enough (aim for 6 suggestions)
                    if len(suggestions) >= 6:
                        break

                # If we still don't have enough suggestions, try to get fallback suggestions
                if len(suggestions) < 3:
                    logger.info(f"Still not enough suggestions ({len(suggestions)}), getting fallback suggestions")
                    fallback_suggestions = await self._get_fallback_suggestions(query, explanation, False)

                    # Add fallback suggestions that aren't already in our list
                    for suggestion in fallback_suggestions:
                        suggestion_name = suggestion.get('name', '').lower()
                        if suggestion_name and suggestion_name not in [s.get('name', '').lower() for s in suggestions]:
                            suggestions.append(suggestion)

                            # Stop when we have enough
                            if len(suggestions) >= 6:
                                break

            # Step 6: Generate response based on explanation and suggestions
            response_prompt = f"""
            You are a car advisor helping customers choose cars.

            Based on the following explanation and car suggestions:

            Explanation: {explanation}

            Suggested cars: {[f"{s.get('brand')} {s.get('model')}" for s in suggestions]}

            User query: {query}

            Please create a conversational response that:
            1. Directly answers the user's question
            2. References the specific cars mentioned in the explanation
            3. Provides a brief comparison between the suggested cars
            4. Is conversational and engaging
            5. Is comprehensive but concise

            VERY IMPORTANT: You MUST ONLY discuss the exact car models listed in the suggestions. Do not mention any other car models.
            """

            # Call the LLM to generate the response
            if self.creative_llm:
                response_obj = await self.creative_llm.ainvoke(response_prompt)
                response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
            else:
                response_obj = await self.llm.ainvoke(response_prompt)
                response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)

            return {
                "response": response,
                "suggestions": suggestions,
                "explanation": explanation
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            # Return a fallback response
            return {
                "response": "I'm sorry, I encountered an error while processing your request. Please try again.",
                "suggestions": [],
                "explanation": "Error processing query."
            }

    def _get_vehicle_type_from_query(self, query):
        """
        Extract the vehicle type from the query

        Args:
            query: User query

        Returns:
            String with vehicle type
        """
        query_lower = query.lower()

        # Check for specific vehicle types
        if any(term in query_lower for term in ["suv", "crossover", "family vehicle"]):
            return "SUV"
        elif any(term in query_lower for term in ["sedan", "saloon", "four door"]):
            return "Sedan"
        elif any(term in query_lower for term in ["coupe", "sports car", "two door"]):
            return "Coupe"
        elif any(term in query_lower for term in ["truck", "pickup"]):
            return "Truck"
        elif any(term in query_lower for term in ["convertible", "cabriolet", "roadster"]):
            return "Convertible"
        elif any(term in query_lower for term in ["wagon", "estate", "touring"]):
            return "Wagon"
        elif any(term in query_lower for term in ["minivan", "mpv", "van"]):
            return "Minivan"
        elif any(term in query_lower for term in ["hatchback", "hot hatch"]):
            return "Hatchback"

        # Default to SUV if nothing specific found
        return "SUV"

    def _estimate_car_price(self, brand, model, vehicle_type):
        """
        Estimate a reasonable price for a car based on brand, model and vehicle type

        Args:
            brand: Car brand
            model: Car model
            vehicle_type: Type of vehicle

        Returns:
            Estimated price
        """
        # Base prices by vehicle type
        base_prices = {
            "SUV": 35000,
            "Sedan": 30000,
            "Coupe": 40000,
            "Truck": 45000,
            "Convertible": 50000,
            "Wagon": 35000,
            "Minivan": 35000,
            "Hatchback": 25000
        }

        # Brand price factors (premium brands cost more)
        brand_factors = {
            "toyota": 1.0,
            "honda": 1.0,
            "ford": 0.95,
            "chevrolet": 0.95,
            "nissan": 0.9,
            "hyundai": 0.85,
            "kia": 0.85,
            "subaru": 1.05,
            "mazda": 0.95,
            "volkswagen": 1.1,
            "bmw": 1.7,
            "mercedes": 1.8,
            "audi": 1.7,
            "lexus": 1.5,
            "porsche": 2.5,
            "tesla": 1.8,
            "volvo": 1.4,
            "jaguar": 1.8,
            "land rover": 1.9,
            "acura": 1.4,
            "infiniti": 1.4,
            "genesis": 1.4,
            "cadillac": 1.6,
            "lincoln": 1.6
        }

        # Get the base price for this vehicle type (default to SUV if not found)
        base_price = base_prices.get(vehicle_type, base_prices["SUV"])

        # Get the brand factor (default to 1.0 if brand not in our list)
        brand_factor = brand_factors.get(brand.lower(), 1.0)

        # Calculate estimated price
        estimated_price = base_price * brand_factor

        # Adjust for specific models that are known to be more expensive
        if "hybrid" in model.lower() or "plug-in" in model.lower():
            estimated_price *= 1.15  # Hybrid models typically cost more
        elif "electric" in model.lower() or "ev" in model.lower():
            estimated_price *= 1.3   # Electric vehicles cost more

        # Round to nearest $1000
        return round(estimated_price / 1000) * 1000

    async def _is_follow_up_question(self, query, conversation_history):
        """
        Determine if the current query is a follow-up to previous conversation

        Args:
            query: Current user query
            conversation_history: List of previous messages

        Returns:
            tuple: (is_follow_up, is_asking_for_more)
            - is_follow_up: True if this is a follow-up question, False otherwise
            - is_asking_for_more: True if the user is asking for more/different options
        """
        if not conversation_history or len(conversation_history) < 2:
            return False, False

        # Check for pronouns and references to previous content
        follow_up_indicators = [
            "it", "they", "them", "those", "these", "that", "this",
            "the car", "the vehicle", "the option", "the model",
            "what about", "how about", "tell me more", "more details",
            "which one", "which of", "between these", "of these",
            "yes", "no", "why", "how", "when", "where", "who",
            "which", "fuel economy", "mpg", "gas mileage", "better",
            "best", "most", "least", "more", "less"
        ]

        # Indicators that the user wants more/different options
        more_options_indicators = [
            "more cars", "more options", "more coupes", "more vehicles",
            "other cars", "other options", "other coupes", "other vehicles",
            "different cars", "different options", "different coupes", "different vehicles",
            "alternative", "alternatives", "something else", "what else",
            "show me more", "tell me more", "additional", "another option"
        ]

        query_lower = query.lower()

        # Check if user is asking for more/different options
        is_asking_for_more = False
        for indicator in more_options_indicators:
            if indicator in query_lower:
                logger.info(f"User is asking for more/different options: '{indicator}' found in query")
                is_asking_for_more = True
                break

        # First check: Does the query explicitly reference a previously mentioned car?
        # Extract car brands from previous assistant messages
        previously_mentioned_cars = []
        if conversation_history:
            for msg in conversation_history:
                if msg.role == "assistant":
                    # Extract car mentions using our existing method
                    extracted_cars = self._extract_cars_from_text(msg.content)
                    for car in extracted_cars:
                        brand = car.get('brand', '').lower()
                        model = car.get('model', '').lower()
                        if brand and brand not in previously_mentioned_cars:
                            previously_mentioned_cars.append(brand)
                        if model and model not in previously_mentioned_cars:
                            previously_mentioned_cars.append(model)

        # Check if query mentions any previously mentioned car
        for car_term in previously_mentioned_cars:
            if car_term in query_lower:
                logger.info(f"Follow-up detected: Query '{query_lower}' mentions previously discussed car term '{car_term}'")
                return True, is_asking_for_more

        # Check if the query asks about car features/attributes of previously mentioned cars
        feature_references = [
            "fuel economy", "mpg", "gas mileage", "fuel consumption",
            "fuel efficiency", "miles per gallon", "range", "performance",
            "price", "cost", "features", "safety", "interior", "comfort",
            "engine", "horsepower", "acceleration", "space", "cargo",
            "reliability", "maintenance", "warranty", "resale value"
        ]

        for feature in feature_references:
            if feature in query_lower:
                logger.info(f"Follow-up detected: Query asks about feature '{feature}' of previously mentioned cars")
                return True, is_asking_for_more

        # Check if query starts with a follow-up indicator
        for indicator in follow_up_indicators:
            if query_lower.startswith(indicator) or f" {indicator} " in query_lower:
                return True, is_asking_for_more

        # Check if query is very short (likely a follow-up)
        if len(query.split()) <= 3:
            return True, is_asking_for_more

        # Check if query seems to be making a comparison between previously mentioned items
        comparison_indicators = ["better", "best", "worse", "worst", "compared", "vs", "versus"]
        for indicator in comparison_indicators:
            if indicator in query_lower:
                return True, is_asking_for_more

        return False, is_asking_for_more

    async def _get_car_data(self, car_id):
        """Get car data by ID"""
        try:
            from app.db.models import Car, CarSpecification
            from app.core.database import SessionLocal
            from sqlalchemy import select

            # Convert car_id to integer if possible
            if isinstance(car_id, str):
                try:
                    car_id = int(car_id)
                except ValueError:
                    logger.error(f"Invalid car ID format: {car_id}, cannot convert to int")
                    return None

            logger.info(f"Fetching car with ID {car_id}")

            # Use database session
            async with SessionLocal() as session:
                # Query the car
                query = select(Car).where(Car.id == car_id)
                result = await session.execute(query)
                car = result.scalars().first()

                if not car:
                    logger.warning(f"Car with ID {car_id} not found in database")
                    return None

                # Get the car specification
                spec_query = select(CarSpecification).where(CarSpecification.car_id == car.id)
                spec_result = await session.execute(spec_query)
                spec = spec_result.scalars().first()

                # Create response dictionary with all detailed fields
                car_dict = {
                    "id": car.id,
                    "name": car.name,
                    "brand": car.brand,
                    "model": car.model,
                    "year": car.year,
                    "price": car.price,
                    "condition": car.condition,
                    "type": car.type,
                    "description": car.description,
                    "image_url": None,
                    "vehicle_style": spec.vehicle_style if spec and spec.vehicle_style else spec.body_type if spec else None,
                    "make": car.brand,
                    "msrp": spec.msrp if spec and spec.msrp else car.price
                }

                # Add specification fields if available
                if spec:
                    car_dict.update({
                        "engine_fuel_type": spec.engine_fuel_type or spec.fuel_type,
                        "engine_hp": spec.engine_hp,
                        "engine_cylinders": spec.engine_cylinders,
                        "transmission_type": spec.transmission_type or spec.transmission,
                        "driven_wheels": spec.driven_wheels,
                        "number_of_doors": spec.number_of_doors,
                        "market_category": spec.market_category,
                        "vehicle_size": spec.vehicle_size,
                        "highway_mpg": spec.highway_mpg,
                        "city_mpg": spec.city_mpg,
                        "popularity": spec.popularity,
                    })

                return car_dict

        except Exception as e:
            logger.error(f"Error getting car data for ID {car_id}: {e}")
            return None

    async def _get_fallback_suggestions(self, query, response_text=None, is_asking_for_more=False):
        """
        Get fallback suggestions based on query content using database information

        Args:
            query: The user query
            response_text: Optional response text to analyze
            is_asking_for_more: Whether the user is asking for more/different options
        """
        suggestions = []
        try:
            # Extract car type from query and response (if provided)
            car_types = {
                "sedan": ["sedan", "4-door", "family car"],
                "suv": ["suv", "crossover", "sport utility vehicle"],
                "truck": ["truck", "pickup", "utility vehicle"],
                "hybrid": ["hybrid", "electric", "eco-friendly"],
                "luxury": ["luxury", "premium", "high-end"],
                "sports": ["sports", "performance", "fast"],
                "coupe": ["coupe", "2-door", "two-door", "sports car"],
                "convertible": ["convertible", "cabriolet", "roadster", "open-top"]
            }

            query_lower = query.lower()
            response_lower = response_text.lower() if response_text else ""

            # Determine car type from query and response
            detected_type = None
            for car_type, keywords in car_types.items():
                for keyword in keywords:
                    if keyword in query_lower or (response_text and keyword in response_lower):
                        detected_type = car_type
                        break
                if detected_type:
                    break

            # If no type was detected, check for specific keywords
            if not detected_type:
                if "coupe" in query_lower:
                    detected_type = "coupe"
                elif "luxury" in query_lower or "premium" in query_lower:
                    detected_type = "luxury"
                elif "sport" in query_lower or "performance" in query_lower:
                    detected_type = "sports"
                elif "suv" in query_lower or "crossover" in query_lower:
                    detected_type = "suv"
                elif "sedan" in query_lower:
                    detected_type = "sedan"
                elif "truck" in query_lower or "pickup" in query_lower:
                    detected_type = "truck"
                elif "convertible" in query_lower:
                    detected_type = "convertible"
                else:
                    # Default to sedan
                    detected_type = "sedan"
                logger.info(f"No car type detected in query, defaulting to {detected_type}")

            # Try multiple search strategies to get cars from database
            search_strategies = [
                # Strategy 1: Search by vehicle style
                {
                    "vehicle_style": detected_type.capitalize() if detected_type not in ["luxury", "hybrid", "sports"] else None,
                    "page": 1,
                    "page_size": 15
                },
                # Strategy 2: Search by market category
                {
                    "search_query": detected_type,
                    "page": 1,
                    "page_size": 15
                },
                # Strategy 3: Search by keywords
                {
                    "search_query": " ".join([detected_type, "popular", "best"]),
                    "page": 1,
                    "page_size": 15
                },
                # Strategy 4: Search by price range based on type
                {
                    "price_min": 0 if detected_type not in ["luxury", "sports"] else 50000,
                    "price_max": 100000 if detected_type not in ["luxury", "sports"] else None,
                    "page": 1,
                    "page_size": 15
                }
            ]

            # If asking for more options, modify search parameters
            if is_asking_for_more:
                for strategy in search_strategies:
                    strategy["page"] = 2
                    strategy["page_size"] = 20

            # Try each search strategy until we get enough results
            cars = []
            for strategy in search_strategies:
                try:
                    search_results = await self.car_service.search_cars_from_csv(**strategy)
                    strategy_cars = search_results.get("items", [])

                    if strategy_cars:
                        # Add new cars to our list, avoiding duplicates
                        for car in strategy_cars:
                            car_id = car.get("id")
                            if car_id and not any(c.get("id") == car_id for c in cars):
                                cars.append(car)

                        # If we have enough cars, stop searching
                        if len(cars) >= 15:
                            break
                except Exception as e:
                    logger.error(f"Error in search strategy: {e}")
                    continue

            # Create a set to track unique brand-model combinations
            seen_brand_models = set()

            # Create suggestions from search results with diversity
            # Track brands and models to ensure diversity
            selected_brands = set()
            selected_models = set()
            max_suggestions = 6  # Always aim for 6 suggestions for better user experience

            # First pass: try to get diverse options by brand and model
            for car in cars:
                try:
                    # Get car id from the 'id' field
                    car_id = car.get("id")

                    # Skip if invalid ID
                    if car_id is None:
                        logger.warning(f"Car has no ID, skipping: {car.get('brand', '')} {car.get('model', '')}")
                        continue

                    # Verify this car exists in the database
                    car_db_data = await self._get_car_data(car_id)
                    if not car_db_data or 'id' not in car_db_data:
                        logger.warning(f"Car ID {car_id} not verified in database")
                        continue

                    # Use the verified database ID and data
                    verified_car_id = car_db_data['id']

                    brand = car_db_data.get('brand', '').strip()
                    model = car_db_data.get('model', '').strip()

                    if not brand or not model:
                        continue

                    # Check for duplicate brand-model
                    brand_model_key = f"{brand.lower()}_{model.lower()}"
                    if brand_model_key in seen_brand_models:
                        continue

                    seen_brand_models.add(brand_model_key)

                    # Skip if we already have this brand-model combination
                    if brand_model_key in selected_models:
                        continue

                    # Limit the number of cars from the same brand to ensure diversity
                    if brand in selected_brands and len(selected_brands) >= 2:
                        continue

                    car_suggestion = CarSuggestion(
                        id=str(verified_car_id),  # Use verified id
                        name=f"{brand} {model}",
                        brand=brand,
                        model=model,
                        price=float(car_db_data.get('msrp', 0)),
                        image_url=car_db_data.get('image_url', None),
                        reasons=[f"Recommended {detected_type.capitalize()} vehicle"]
                    )

                    # Convert to dict using model_dump
                    suggestion_dict = car_suggestion.model_dump()

                    suggestions.append(suggestion_dict)
                    selected_brands.add(brand)
                    selected_models.add(brand_model_key)
                    logger.info(f"Added fallback suggestion: {brand} {model} (ID: {verified_car_id})")

                    # Stop once we have enough diverse suggestions
                    if len(suggestions) >= max_suggestions:
                        break

                except Exception as e:
                    logger.error(f"Error creating fallback suggestion: {e}")
                    continue

            # If we don't have enough suggestions, add more from the remaining cars
            # For the second pass, we'll be less strict about brand diversity
            if len(suggestions) < 3:
                logger.info(f"Not enough diverse suggestions ({len(suggestions)}), adding more")

                # Reset the seen brands for the second pass to allow more from the same brand
                # but keep track of exact models we've already added
                for car in cars:
                    if len(suggestions) >= 3:
                        break

                    try:
                        # Get car id from the 'id' field
                        car_id = car.get("id")

                        # Skip if invalid ID
                        if car_id is None:
                            continue

                        # Verify this car exists in the database
                        car_db_data = await self._get_car_data(car_id)
                        if not car_db_data or 'id' not in car_db_data:
                            continue

                        # Use the verified database ID and data
                        verified_car_id = car_db_data['id']

                        brand = car_db_data.get('brand', '').strip()
                        model = car_db_data.get('model', '').strip()

                        if not brand or not model:
                            continue

                        # Check for duplicate brand-model
                        brand_model_key = f"{brand.lower()}_{model.lower()}"
                        if brand_model_key in seen_brand_models:
                            continue

                        seen_brand_models.add(brand_model_key)

                        car_suggestion = CarSuggestion(
                            id=str(verified_car_id),  # Use verified id
                            name=f"{brand} {model}",
                            brand=brand,
                            model=model,
                            price=float(car_db_data.get('msrp', 0)),
                            image_url=car_db_data.get('image_url', None),
                            reasons=[f"Recommended {detected_type.capitalize()} vehicle"]
                        )

                        # Convert to dict using model_dump
                        suggestion_dict = car_suggestion.model_dump()

                        suggestions.append(suggestion_dict)
                        logger.info(f"Added additional fallback suggestion: {brand} {model} (ID: {verified_car_id})")
                    except Exception as e:
                        logger.error(f"Error creating additional fallback suggestion: {e}")
                        continue

            logger.info(f"Created {len(suggestions)} fallback suggestions for {detected_type} category")
        except Exception as e:
            logger.error(f"Error getting fallback suggestions: {e}")

        return suggestions

    def _extract_cars_from_text(self, text):
        """
        Extract car mentions from text

        Args:
            text: The text to extract car mentions from

        Returns:
            List of dictionaries with car brand and model
        """
        try:
            # Common car brands to look for
            car_brands = ["Toyota", "Honda", "Ford", "Chevrolet", "BMW", "Mercedes",
                         "Audi", "Lexus", "Nissan", "Hyundai", "Kia", "Subaru",
                         "Volkswagen", "Mazda", "Jeep", "Tesla", "Volvo", "Porsche",
                         "Acura", "Cadillac", "Land Rover", "Range Rover", "Infiniti",
                         "Mitsubishi", "Dodge", "Chrysler", "Buick", "GMC", "Lincoln"]

            extracted_cars = []

            # First attempt: Process with a more flexible pattern to catch various formats
            # Look for format like "Brand Model" with various separators
            for brand in car_brands:
                # Multiple regex patterns to match different formats:
                # 1. Brand followed by model with space
                # 2. Brand followed by model with hyphen or other separators
                # 3. Brand followed by model with year or other descriptors
                patterns = [
                    fr"{re.escape(brand)}\s+([A-Za-z0-9][\w\-\s]+?)(?:\s+\d|\s+is|\s+has|\s+offers|\s+comes|\s+would|\s+might|\s+could|\W|\Z)",
                    fr"{re.escape(brand)}[- ]([A-Za-z0-9][\w\-]+)",
                    fr"the\s+{re.escape(brand)}\s+([A-Za-z0-9][\w\-]+)"
                ]

                for pattern in patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        model = match.strip()
                        # Skip empty or obviously incorrect models
                        if not model or len(model) < 2 or model.lower() in ['is', 'has', 'and', 'or', 'the', 'a']:
                            continue

                        # Clean up model names that might have trailing punctuation or words
                        model = re.sub(r'\s+(is|has|offers|comes|with|would|could|might|may).*$', '', model, flags=re.IGNORECASE)
                        model = re.sub(r'[.,;:!?].*$', '', model)
                        model = model.strip()

                        car = {
                            'brand': brand,
                            'model': model,
                            'name': f"{brand} {model}",
                            'price': 0  # Default price
                        }

                        # Check if this car is already in the list (case insensitive)
                        if not any(c['name'].lower() == car['name'].lower() for c in extracted_cars):
                            extracted_cars.append(car)

            # If we found cars, return them
            if extracted_cars:
                return extracted_cars

            # Second attempt: Look for mentions of "car" or "vehicle" and try to extract nearby brand/model
            car_mentions = re.finditer(r'(car|vehicle|automobile|model)(?:\s+(?:is|called|named)\s+([A-Za-z]+\s+[A-Za-z0-9\-]+))?', text, re.IGNORECASE)

            for mention in car_mentions:
                sentence_start = max(0, mention.start() - 50)
                sentence_end = min(len(text), mention.end() + 50)
                sentence = text[sentence_start:sentence_end]

                # Look for brand names in this sentence
                for brand in car_brands:
                    if re.search(fr'\b{re.escape(brand)}\b', sentence, re.IGNORECASE):
                        # Extract words after the brand that might be a model
                        brand_matches = re.search(fr'\b{re.escape(brand)}\b\s+([A-Za-z0-9][\w\-]+)', sentence, re.IGNORECASE)
                        if brand_matches:
                            model = brand_matches.group(1).strip()
                            if model and not any(c['name'].lower() == f"{brand} {model}".lower() for c in extracted_cars):
                                extracted_cars.append({
                                    'brand': brand,
                                    'model': model,
                                    'name': f"{brand} {model}",
                                    'price': 0
                                })

            return extracted_cars
        except Exception as e:
            logger.error(f"Error extracting cars from text: {e}")
            return []

    async def _get_car_data_by_name(self, brand, model):
        """
        Get car data by brand and model from database

        Args:
            brand: Car brand
            model: Car model

        Returns:
            Car data dictionary or None if not found
        """
        try:
            # Log the search request
            logger.info(f"Searching for car: {brand} {model}")

            # Clean and normalize brand and model names
            brand = str(brand).strip()
            model = str(model).strip()

            # Try direct database lookup first
            from sqlalchemy import select, or_
            from app.db.models import Car, CarSpecification
            from app.core.database import SessionLocal

            # Create model variations to improve search results
            model_variations = [
                model,
                model.replace(' ', ''),
                model.replace(' ', '-'),
                model.replace('-', ' '),
                model.replace('series', '').strip(),
                model.replace('class', '').strip()
            ]

            # Use a session directly
            async with SessionLocal() as session:
                # Try exact match first
                query = select(Car).where(
                    Car.brand.ilike(f"%{brand}%") &
                    or_(*[Car.model.ilike(f"%{var}%") for var in model_variations])
                ).limit(5)

                result = await session.execute(query)
                cars = result.scalars().all()

                if cars:
                    # Found cars in database
                    car = cars[0]  # Take the first match
                    logger.info(f"Found car in database: {car.brand} {car.model}")

                    # Get specifications
                    spec_query = select(CarSpecification).where(CarSpecification.car_id == car.id)
                    spec_result = await session.execute(spec_query)
                    spec = spec_result.scalars().first()

                    # Create car data dictionary
                    car_data = {
                        "id": car.id,
                        "Make": car.brand,
                        "Model": car.model,
                        "Year": car.year,
                        "MSRP": spec.msrp if spec and spec.msrp else car.price,
                        "Vehicle Style": spec.vehicle_style if spec and spec.vehicle_style else spec.body_type if spec else None,
                        "Market Category": spec.market_category if spec and spec.market_category else "Standard"
                    }

                    # Add additional fields from specification if available
                    if spec:
                        car_data.update({
                            "Engine Fuel Type": spec.engine_fuel_type or spec.fuel_type,
                            "Engine HP": spec.engine_hp,
                            "Engine Cylinders": spec.engine_cylinders,
                            "Transmission Type": spec.transmission_type or spec.transmission,
                            "Driven_Wheels": spec.driven_wheels,
                            "Number of Doors": spec.number_of_doors,
                            "Vehicle Size": spec.vehicle_size,
                            "highway MPG": spec.highway_mpg,
                            "city mpg": spec.city_mpg,
                            "Popularity": spec.popularity
                        })

                    return car_data

            # If direct database lookup failed, try search_cars API
            logger.info(f"Direct database lookup failed, trying search_cars_from_csv API")

            # Handle brand name variations
            brand_variations = [brand]

            # Special case for Mercedes vs Mercedes-Benz
            if brand.lower() == "mercedes":
                brand_variations.extend(["mercedes-benz", "mercedesbenz"])
            elif brand.lower() == "mercedes benz":
                brand_variations.extend(["mercedes-benz", "mercedesbenz", "mercedes"])
            elif brand.lower() == "mercedes-benz":
                brand_variations.extend(["mercedes", "mercedesbenz"])

            # Add retry logic
            max_retries = 3
            retry_delay = 1  # seconds

            for attempt in range(max_retries):
                try:
                    # Try with make/model search
                    for brand_var in brand_variations:
                        for model_var in model_variations:
                            search_params = {
                                "make": brand_var,
                                "model": model_var,
                                "page": 1,
                                "page_size": 10,
                                "partial_match": True  # Use partial matching
                            }

                            # Get cars
                            search_results = await self.car_service.search_cars_from_csv(**search_params)
                            cars = search_results.get("items", [])

                            if cars and len(cars) > 0:
                                logger.info(f"Found car with search API: {cars[0].get('Make', '')} {cars[0].get('Model', '')}")
                                return cars[0]

                    # If not found, try with just the brand
                    for brand_var in brand_variations:
                        search_params = {
                            "make": brand_var,
                            "page": 1,
                            "page_size": 10
                        }

                        search_results = await self.car_service.search_cars_from_csv(**search_params)
                        cars = search_results.get("items", [])

                        if cars and len(cars) > 0:
                            logger.info(f"Found car with brand-only search: {cars[0].get('Make', '')} {cars[0].get('Model', '')}")
                            return cars[0]

                    # If we get here, retry
                    if attempt < max_retries - 1:
                        logger.info(f"Retry attempt {attempt + 1} for {brand} {model}")
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.warning(f"No car found for {brand} {model} after {max_retries} attempts")
                        return None

                except Exception as e:
                    logger.error(f"Error in search attempt {attempt + 1} for {brand} {model}: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                    else:
                        raise

        except Exception as e:
            logger.error(f"Error getting car data by name: {e}")
            return None

    async def _generate_explanation(self, suggestions, query, conversation_history=None):
        """
        Generate an explanation for why these cars were suggested

        Args:
            suggestions: List of car suggestions
            query: Current user query
            conversation_history: Optional list of previous messages
        """
        try:
            if not suggestions:
                return "No specific car suggestions were found based on your query."

            # Create a prompt for explanation
            car_descriptions = []
            for car in suggestions:
                brand = car.get('brand', '')
                model = car.get('model', '')
                price = car.get('price', 0)
                if brand and model:
                    car_descriptions.append(f"{brand} {model} (${price:,.2f})")

            if not car_descriptions:
                return "No specific car suggestions were found."

            # Include conversation history context if available
            conversation_context = ""
            if conversation_history and len(conversation_history) > 0:
                conversation_context = "Previous conversation:\n"
                for msg in conversation_history[-3:]:  # Only use last 3 messages for brevity
                    role = "User" if msg.role == "user" else "Assistant"
                    conversation_context += f"{role}: {msg.content}\n"

            explanation_prompt = f"""
            {conversation_context}

            The user's current question: "{query}"

            Based on this conversation and query, the following cars were suggested:
            {', '.join(car_descriptions)}

            Provide a personalized explanation for why these cars might be suitable for the user's needs.
            Your explanation should:
            1. Be conversational and natural
            2. Reference any specific requirements or preferences mentioned in the conversation
            3. Highlight key features of the suggested cars that match the user's needs
            4. Vary in style to sound more human-like and less repetitive
            5. ALWAYS discuss MULTIPLE car options (at least 3 different cars)
            6. Compare the different car options to help the user make an informed decision
            7. NEVER focus on just one car - always provide multiple options for comparison

            If this is a follow-up question, make sure your explanation maintains continuity with the previous conversation.
            """

            # Determine if this is likely a follow-up question based on conversation history
            is_follow_up = False
            if conversation_history and len(conversation_history) >= 2:
                is_follow_up = True

            # Use creative LLM for follow-up questions to get more varied responses
            # This helps prevent the chatbot from sounding repetitive
            if is_follow_up and self.creative_llm:
                response = await self.creative_llm.ainvoke(explanation_prompt)
            else:
                response = await self.llm.ainvoke(explanation_prompt)

            return response.content
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return "These cars match your search criteria."

    async def _enhance_query(self, query, conversation_history=None):
        """
        Enhance the query to make vector search more effective and diverse.

        This method analyzes the query and conversation history to create a more
        effective search query that will yield diverse and relevant results.

        Args:
            query: The original user query
            conversation_history: List of previous messages in the conversation

        Returns:
            An enhanced query string
        """
        try:
            import random

            # Define car-related categories and terms for query enhancement
            car_categories = {
                "vehicle_types": ["sedan", "suv", "truck", "crossover", "hatchback", "coupe", "convertible", "minivan", "wagon"],
                "price_ranges": ["budget", "affordable", "mid-range", "luxury", "premium", "expensive"],
                "features": ["spacious", "fuel-efficient", "hybrid", "electric", "powerful", "sporty", "comfortable", "safety"],
                "brands": ["Toyota", "Honda", "Ford", "Chevrolet", "BMW", "Mercedes", "Audi", "Lexus", "Nissan",
                           "Hyundai", "Kia", "Subaru", "Volkswagen", "Mazda", "Jeep", "Tesla"]
            }

            # Initialize the enhanced query with the original query
            enhanced_query = query.strip()
            original_query_lower = query.lower()

            # If this is a very short query, add more context
            if len(enhanced_query.split()) <= 3:
                enhanced_query = f"car recommendation for {enhanced_query}"

            # Extract any vehicle types mentioned in the query
            mentioned_vehicle_types = []
            for vtype in car_categories["vehicle_types"]:
                if vtype in original_query_lower:
                    mentioned_vehicle_types.append(vtype)

            # Extract any brands mentioned in the query
            mentioned_brands = []
            for brand in car_categories["brands"]:
                if brand.lower() in original_query_lower:
                    mentioned_brands.append(brand)

            # Extract any price ranges mentioned in the query
            mentioned_price_ranges = []
            for price in car_categories["price_ranges"]:
                if price in original_query_lower:
                    mentioned_price_ranges.append(price)

            # Extract any features mentioned in the query
            mentioned_features = []
            for feature in car_categories["features"]:
                if feature in original_query_lower:
                    mentioned_features.append(feature)

            # Add diversity if the query is too specific to one category
            # This helps prevent getting stuck in a single vehicle type or brand

            # If only one vehicle type is mentioned, suggest a related one
            if len(mentioned_vehicle_types) == 1:
                related_types = {
                    "sedan": ["compact", "midsize"],
                    "suv": ["crossover", "compact suv"],
                    "truck": ["pickup", "utility vehicle"],
                    "crossover": ["small suv", "compact suv"],
                    "hatchback": ["compact", "small car"],
                    "coupe": ["sports car", "performance car"],
                    "convertible": ["roadster", "sports car"],
                    "minivan": ["family vehicle", "passenger van"],
                    "wagon": ["estate", "touring"]
                }
                if mentioned_vehicle_types[0] in related_types:
                    alt_type = random.choice(related_types[mentioned_vehicle_types[0]])
                    enhanced_query += f" or {alt_type}"

            # If no vehicle type is mentioned, randomly add one based on the query context
            if not mentioned_vehicle_types:
                # Try to infer vehicle type from context
                if "family" in original_query_lower or "kids" in original_query_lower or "spacious" in original_query_lower:
                    suggested_types = ["suv", "minivan", "crossover"]
                elif "efficient" in original_query_lower or "commuting" in original_query_lower or "economy" in original_query_lower:
                    suggested_types = ["sedan", "hatchback", "hybrid"]
                elif "luxury" in original_query_lower or "premium" in original_query_lower:
                    suggested_types = ["luxury sedan", "luxury suv", "premium vehicle"]
                elif "off-road" in original_query_lower or "adventure" in original_query_lower:
                    suggested_types = ["suv", "truck", "jeep"]
                elif "sport" in original_query_lower or "performance" in original_query_lower:
                    suggested_types = ["sports car", "performance vehicle", "coupe"]
                elif "coupe" in original_query_lower or "2-door" in original_query_lower or "two-door" in original_query_lower:
                    suggested_types = ["coupe", "sports car", "performance car"]
                else:
                    # Default to a random selection of common vehicle types
                    suggested_types = ["sedan", "suv", "crossover", "hatchback"]

                # Add a suggested vehicle type to enhance the query
                enhanced_query += f" {random.choice(suggested_types)}"

            # If specific brands are mentioned, ensure we include diversity
            # but respect the user's preferences
            if len(mentioned_brands) == 1:
                # If only one brand is mentioned, add an "or similar" term to encourage diversity
                enhanced_query += " or similar vehicles"

            # If no brands are mentioned, we don't add random brands to avoid misleading results

            # Add conversation context if available
            if conversation_history and len(conversation_history) > 0:
                # Extract key terms from previous messages
                context_terms = set()
                for msg in conversation_history[-3:]:  # Only use the last 3 messages
                    content = msg.content.lower()
                    # Extract vehicle types
                    for vtype in car_categories["vehicle_types"]:
                        if vtype in content and vtype not in mentioned_vehicle_types:
                            context_terms.add(vtype)
                    # Extract features
                    for feature in car_categories["features"]:
                        if feature in content and feature not in mentioned_features:
                            context_terms.add(feature)

                # Add up to 2 context terms for diversity without overloading the query
                if context_terms:
                    terms_to_add = random.sample(list(context_terms), min(2, len(context_terms)))
                    enhanced_query += " " + " ".join(terms_to_add)

            logger.info(f"Enhanced query from '{query}' to '{enhanced_query}'")
            return enhanced_query

        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            # If enhancement fails, return original query
            return query

    def _create_filters_from_query(self, query):
        """
        Create filters based on the query

        Args:
            query: The original user query

        Returns:
            A dictionary of filters
        """
        filters = {}
        try:
            # Extract vehicle type from query
            vehicle_types = {
                "sedan": ["sedan", "4-door", "family car"],
                "suv": ["suv", "crossover", "sport utility vehicle"],
                "truck": ["truck", "pickup", "utility vehicle"],
                "hybrid": ["hybrid", "electric", "eco-friendly"],
                "luxury": ["luxury", "premium", "high-end"],
                "sports": ["sports", "performance", "fast"],
                "coupe": ["coupe", "2-door", "two-door", "sports car"]
            }
            for vtype, keywords in vehicle_types.items():
                for keyword in keywords:
                    if keyword in query.lower():
                        filters['vehicle_style'] = vtype
                        break

            # Extract price range from query
            price_ranges = ["budget", "affordable", "mid-range", "luxury", "premium", "expensive"]
            for price in price_ranges:
                if price in query.lower():
                    filters['price'] = price
                    break

            # Extract features from query
            features = ["spacious", "fuel-efficient", "powerful", "sporty", "comfortable", "safety"]
            for feature in features:
                if feature in query.lower():
                    filters['features'] = feature
                    break

            # Extract brand from query
            car_brands = ["Toyota", "Honda", "Ford", "Chevrolet", "BMW", "Mercedes",
                         "Audi", "Lexus", "Nissan", "Hyundai", "Kia", "Subaru",
                         "Volkswagen", "Mazda", "Jeep", "Tesla", "Volvo"]
            for brand in car_brands:
                if brand.lower() in query.lower():
                    filters['brand'] = brand
                    break

            return filters
        except Exception as e:
            logger.error(f"Error creating filters from query: {e}")
            return {}

    async def _generate_response_with_llama_index(self, query, docs, conversation_history=None):
        """
        Sử dụng LLM trực tiếp thay vì ResponseSynthesizer để tránh lỗi import

        Args:
            query: Câu truy vấn người dùng
            docs: Danh sách các tài liệu liên quan
            conversation_history: Lịch sử hội thoại

        Returns:
            Phản hồi được tổng hợp
        """
        try:
            # Nếu không có tài liệu, trả về thông báo mặc định
            if not docs:
                return "I couldn't find information matching your request."

            # Tạo context từ tài liệu
            context_text = "\n\n".join([
                doc.page_content for doc in docs
                if hasattr(doc, 'page_content') and doc.page_content
            ])

            # Tạo context từ lịch sử hội thoại
            conv_context = ""
            if conversation_history and len(conversation_history) > 0:
                conv_context = "Conversation history:\n"
                for msg in conversation_history[-3:]:  # Chỉ sử dụng 3 tin nhắn gần nhất
                    role = "User" if msg.role == "user" else "Assistant"
                    conv_context += f"{role}: {msg.content}\n"

            # Tạo prompt tùy chỉnh
            prompt = f"""
            You are a car advisor helping customers choose suitable vehicles.
            Use the following information to answer the question.

            {conv_context}

            Relevant information:
            {context_text}

            Question: {query}

            Please note:
            1. ALWAYS discuss MULTIPLE car options (at least 3 different cars) in your response.
            2. Compare the benefits and limitations of the suggested cars.
            3. If this is a follow-up question, refer to cars mentioned in the conversation history.
            4. Ensure your response is helpful and personalized.
            5. NEVER recommend just one car - always provide multiple options for comparison.

            Response:
            """

            # Sử dụng LLM đã khởi tạo cho class RAGService
            if self.creative_llm:
                response_obj = await self.creative_llm.ainvoke(prompt)
                if hasattr(response_obj, 'content'):
                    return response_obj.content
                return str(response_obj)
            elif self.llm:
                response_obj = await self.llm.ainvoke(prompt)
                if hasattr(response_obj, 'content'):
                    return response_obj.content
                return str(response_obj)
            else:
                return "Unable to generate a response because the LLM hasn't been initialized."

        except Exception as e:
            logger.error(f"Error generating response with simple LLM: {e}")
            # Fallback sang phương thức gốc
            return None

    async def _ensure_suggestion_diversity(self, suggestions):
        """
        Ensure suggestions have a good diversity of brands and models

        Args:
            suggestions: List of car suggestions

        Returns:
            List of car suggestions with ensured diversity
        """
        try:
            if not suggestions:
                return []

            # Required parameters
            max_per_brand = 2  # Maximum cars from same brand
            min_total = 3      # Minimum total suggestions to return
            max_total = 6      # Maximum total suggestions to return

            # First, let's deduplicate completely identical brand+model combinations
            # This is to fix cases where we get multiple identical entries (like multiple Toyota 4Runner)
            deduplicated_suggestions = []
            seen_brand_models = set()

            for suggestion in suggestions:
                brand = suggestion.get('brand', '').strip()
                model = suggestion.get('model', '').strip()

                if not brand or not model:
                    continue

                brand_model_key = f"{brand.lower()}_{model.lower()}"

                if brand_model_key not in seen_brand_models:
                    deduplicated_suggestions.append(suggestion)
                    seen_brand_models.add(brand_model_key)
                else:
                    logger.info(f"Removing duplicate suggestion: {brand} {model}")

            # Continue with the deduplicated list
            suggestions = deduplicated_suggestions

            # Step 1: Group suggestions by brand
            brand_groups = {}
            for suggestion in suggestions:
                brand = suggestion.get('brand', '')
                if not brand:
                    continue

                if brand not in brand_groups:
                    brand_groups[brand] = []

                brand_groups[brand].append(suggestion)

            # Step 2: Select diverse suggestions
            diverse_suggestions = []
            used_models = set()

            # First, get one car from each brand
            for brand, brand_suggestions in brand_groups.items():
                if len(diverse_suggestions) >= max_total:
                    break

                # Sort by price (ascending) to prioritize more affordable options
                brand_suggestions.sort(key=lambda s: float(s.get('price', 0)))

                # Add the first suggestion if model not used
                for suggestion in brand_suggestions:
                    model = suggestion.get('model', '')
                    if model and model not in used_models:
                        diverse_suggestions.append(suggestion)
                        used_models.add(model)
                        logger.info(f"Added diverse suggestion (first pass): {brand} {model}")
                        break

            # If we don't have enough, add a second car from each brand
            if len(diverse_suggestions) < min_total:
                for brand, brand_suggestions in brand_groups.items():
                    if len(diverse_suggestions) >= max_total:
                        break

                    # Count how many we already have from this brand
                    brand_count = sum(1 for s in diverse_suggestions if s.get('brand', '') == brand)

                    # Add another if we're below the limit
                    if brand_count < max_per_brand:
                        for suggestion in brand_suggestions:
                            model = suggestion.get('model', '')
                            if model and model not in used_models:
                                diverse_suggestions.append(suggestion)
                                used_models.add(model)
                                logger.info(f"Added diverse suggestion (second pass): {brand} {model}")
                                break

            # Step 3: If we still don't have enough suggestions, add more from original list
            if len(diverse_suggestions) < min_total:
                for suggestion in suggestions:
                    brand = suggestion.get('brand', '')
                    model = suggestion.get('model', '')
                    if not brand or not model:
                        continue

                    brand_model_key = f"{brand.lower()}_{model.lower()}"

                    # Check if this exact car is already in diverse_suggestions
                    if not any(s.get('brand', '').lower() == brand.lower() and
                               s.get('model', '').lower() == model.lower()
                               for s in diverse_suggestions):
                        if len(diverse_suggestions) < min_total:
                            diverse_suggestions.append(suggestion)
                            logger.info(f"Added fallback suggestion: {brand} {model}")

            # Step 4: Add helpful metadata about vehicle types if missing
            for suggestion in diverse_suggestions:
                if 'reasons' not in suggestion or not suggestion['reasons']:
                    car_id = suggestion.get('id')
                    if car_id:
                        try:
                            car_data = await self._get_car_data(car_id)
                            if car_data:
                                vehicle_style = car_data.get('vehicle_style') or car_data.get('vehicle_type', 'vehicle')
                                suggestion['reasons'] = [f"Recommended {vehicle_style}"]
                        except Exception as e:
                            logger.error(f"Error adding car type to suggestion: {e}")

            logger.info(f"Diversified suggestions: {[f'{s.get('brand')} {s.get('model')}' for s in diverse_suggestions]}")
            return diverse_suggestions

        except Exception as e:
            logger.error(f"Error ensuring suggestion diversity: {e}")
            return suggestions[:min(6, len(suggestions))]

    async def _generate_final_response(self, initial_response, suggestions, explanation, query, conversation_history=None):
        """
        Generate a final response combining the initial response, suggestions, and explanation

        Args:
            initial_response: The initial response generated by llama-index
            suggestions: List of car suggestions
            explanation: The explanation for why these cars were suggested
            query: The original user query
            conversation_history: List of previous messages in the conversation

        Returns:
            A final response combining the initial response, suggestions, and explanation
        """
        try:
            # Check if this is a follow-up question about previously mentioned cars
            is_follow_up = await self._is_follow_up_question(query, conversation_history)

            # If this is a follow-up question, identify previously mentioned cars to maintain continuity
            previous_car_suggestions = []
            if is_follow_up and conversation_history and len(conversation_history) >= 2:
                # Look at the previous exchanges (up to last 2 assistant messages)
                assistant_messages = [msg for msg in conversation_history if msg.role == "assistant"][-2:]

                for msg in assistant_messages:
                    # Extract car mentions from previous assistant messages
                    extracted_cars = self._extract_cars_from_text(msg.content)

                    # For each extracted car, check if we can find it in our database
                    for car in extracted_cars:
                        brand = car.get('brand', '')
                        model = car.get('model', '')

                        # Look up this car in the database
                        car_data = await self._get_car_data_by_name(brand, model)

                        if car_data and 'id' in car_data:
                            # We found the car in the database
                            car_id = car_data.get('id')

                            # Check if this car is already in our suggestions (avoid duplicates)
                            existing_ids = [s.get('id') for s in suggestions if s.get('id')]
                            if str(car_id) not in existing_ids:
                                # Get detailed data for this car
                                car_db_data = await self._get_car_data(car_id)

                                if car_db_data and 'id' in car_db_data:
                                    # Create a new suggestion for this previously mentioned car
                                    car_suggestion = CarSuggestion(
                                        id=str(car_id),
                                        name=f"{brand} {model}",
                                        brand=brand,
                                        model=model,
                                        price=float(car_db_data.get('msrp', car_db_data.get('price', 0))),
                                        image_url=car_db_data.get('image_url', None),
                                        reasons=[f"Previously discussed vehicle"]
                                    )

                                    # Convert to dict using model_dump
                                    suggestion_dict = car_suggestion.model_dump()
                                    previous_car_suggestions.append(suggestion_dict)
                                    logger.info(f"Added previously mentioned car to suggestions: {brand} {model}")

            # For follow-up questions, prioritize previously mentioned cars
            # If we have previous car suggestions, replace current suggestions with them
            if is_follow_up and previous_car_suggestions:
                # First check if all previous cars are already in suggestions
                current_ids = set(s.get('id') for s in suggestions if s.get('id'))
                previous_ids = set(s.get('id') for s in previous_car_suggestions if s.get('id'))

                # If there's no overlap, we should use the previous cars instead
                if not current_ids.intersection(previous_ids):
                    logger.info(f"Replacing current suggestions with previously mentioned cars for follow-up question")
                    suggestions = previous_car_suggestions
                else:
                    # Merge the suggestions, prioritizing previous cars
                    merged_suggestions = previous_car_suggestions.copy()

                    # Add current suggestions not already in the list
                    merged_ids = set(s.get('id') for s in merged_suggestions if s.get('id'))
                    for suggestion in suggestions:
                        if suggestion.get('id') not in merged_ids:
                            merged_suggestions.append(suggestion)
                            merged_ids.add(suggestion.get('id'))

                    suggestions = merged_suggestions
                    logger.info(f"Merged previous and current car suggestions for follow-up question")

            # Create detailed information about each suggestion
            suggestion_details = []
            suggestion_brands_models = set()  # Track brand+model combinations for lookup

            # Track the cars being suggested for later verification
            for suggestion in suggestions:
                brand = suggestion.get('brand', '')
                model = suggestion.get('model', '')
                price = suggestion.get('price', 0)
                vehicle_style = suggestion.get('vehicle_style', '') or \
                               suggestion.get('vehicle_type', '') or \
                               (suggestion.get('reasons', [''])[0].replace('Recommended ', '') if suggestion.get('reasons') else '')

                detail = f"{brand} {model} (${price:,.2f})"
                if vehicle_style and vehicle_style.lower() != 'vehicle':
                    detail += f" - {vehicle_style}"

                suggestion_details.append(detail)
                suggestion_brands_models.add(f"{brand.lower()}_{model.lower()}")

            # Format conversation history if available
            conversation_context = ""
            if conversation_history and len(conversation_history) > 0:
                conversation_context = "Previous conversation:\n"
                for msg in conversation_history[-3:]:  # Only use last 3 messages for brevity
                    role = "User" if msg.role == "user" else "Assistant"
                    conversation_context += f"{role}: {msg.content}\n"

            # Add explicit follow-up handling instructions to the prompt
            follow_up_instructions = ""
            if is_follow_up:
                follow_up_instructions = """
                IMPORTANT: This is a follow-up question about previously mentioned cars.
                You MUST ONLY discuss the exact same cars that were mentioned in previous messages.
                Do NOT introduce new car models that weren't previously discussed.
                Focus your response specifically on answering the follow-up question about the previously mentioned cars.
                """

            # Create a prompt for the final response
            final_prompt = f"""
            You are a car advisor helping customers choose cars.

            {conversation_context}

            User's current question: "{query}"

            I have already prepared an initial response to the user:
            "{initial_response}"

            I have also identified these specific car suggestions that match the user's needs:
            {', '.join(suggestion_details)}

            Additional explanation for the suggestions:
            {explanation}

            {follow_up_instructions}

            Please create a single, coherent response that:
            1. Directly answers the user's question
            2. Incorporates the initial response insights
            3. Specifically mentions ONLY the cars in the suggestions list by name (do not introduce new cars)
            4. Provides a brief comparison between the suggested cars
            5. Is conversational and engaging
            6. Is comprehensive but concise

            VERY IMPORTANT: You MUST ONLY discuss the exact car models listed in the suggestions I provided. Do not mention any other car models that aren't in the suggestions list.
            For clarity, please ONLY discuss these exact models: {', '.join([f"{s.get('brand')} {s.get('model')}" for s in suggestions])}

            Important: The response should feel like a natural conversation, not a list or collection of separate pieces.
            Focus on the most relevant details for the user's specific question.
            """

            # Call the LLM to generate the final response
            # Choose the creative LLM for more natural-sounding responses
            if self.creative_llm:
                response_obj = await self.creative_llm.ainvoke(final_prompt)
                final_response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
            elif self.llm:
                response_obj = await self.llm.ainvoke(final_prompt)
                final_response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
            else:
                # Fallback if no LLM is available
                logger.warning("No LLM available for final response generation, using concatenation instead")
                final_response = f"{initial_response}\n\n"
                if suggestions:
                    final_response += "Based on your needs, I recommend considering these options:\n"
                    for detail in suggestion_details:
                        final_response += f"- {detail}\n"
                final_response += f"\n{explanation}"

            # Verify the final response only mentions cars that are in our suggestions
            # Extract car mentions from the final response
            extracted_cars = self._extract_cars_from_text(final_response)
            logger.info(f"Cars extracted from final response: {extracted_cars}")

            # Check if any extracted cars are not in our suggestions
            missing_cars = []
            for car in extracted_cars:
                brand = car.get('brand', '')
                model = car.get('model', '')
                car_key = f"{brand.lower()}_{model.lower()}"

                if car_key not in suggestion_brands_models:
                    missing_cars.append(f"{brand} {model}")
                    logger.warning(f"Car mentioned in response but not in suggestions: {brand} {model}")

            # If there are missing cars, we need to update either the suggestions or the response
            if missing_cars:
                logger.warning(f"Found {len(missing_cars)} cars in response not in suggestions: {missing_cars}")

                # Try to find these cars in the database and add to suggestions
                updated_suggestions = list(suggestions)  # Create a copy

                for car_name in missing_cars:
                    parts = car_name.split(" ", 1)
                    if len(parts) == 2:
                        brand, model = parts

                        # Look up this car in the database
                        car_data = await self._get_car_data_by_name(brand, model)

                        if car_data and 'id' in car_data:
                            # Create a new suggestion for this car
                            car_suggestion = CarSuggestion(
                                id=str(car_data.get('id')),
                                name=f"{brand} {model}",
                                brand=brand,
                                model=model,
                                price=float(car_data.get('msrp', car_data.get('price', 0))),
                                image_url=car_data.get('image_url', None),
                                reasons=[f"Referenced in response"]
                            )

                            # Add to updated suggestions
                            suggestion_dict = car_suggestion.model_dump()
                            updated_suggestions.append(suggestion_dict)
                            logger.info(f"Added missing car to suggestions: {brand} {model}")

                # If we found and added the missing cars, update our return value
                if len(updated_suggestions) > len(suggestions):
                    logger.info(f"Updated suggestions to include cars mentioned in response")
                    suggestions.clear()
                    suggestions.extend(updated_suggestions)

            logger.info("Generated final response with LLM that includes specific car suggestions")
            return final_response, suggestions

        except Exception as e:
            logger.error(f"Error generating final response: {e}")
            # Fallback to initial response if there's an error
            return initial_response, []

    async def _get_emergency_suggestions(self, query, min_needed):
        """
        Get emergency suggestions to ensure we have enough car options

        Args:
            query: The original user query
            min_needed: The minimum number of car suggestions needed

        Returns:
            List of emergency car suggestions
        """
        try:
            # Initialize empty suggestions list
            suggestions = []

            # Analyze query to determine vehicle type
            query_lower = query.lower()

            # Detect vehicle type from query
            vehicle_styles = []
            if any(term in query_lower for term in ["suv", "family vehicle", "crossover", "family car"]):
                vehicle_styles = ["SUV", "Crossover"]
            elif any(term in query_lower for term in ["sedan", "saloon", "four door"]):
                vehicle_styles = ["Sedan", "4-door"]
            elif any(term in query_lower for term in ["truck", "pickup"]):
                vehicle_styles = ["Truck", "Pickup"]
            elif any(term in query_lower for term in ["sport", "performance", "fast"]):
                vehicle_styles = ["Sports", "Coupe"]
            elif any(term in query_lower for term in ["luxury", "premium"]):
                vehicle_styles = ["Luxury"]
            else:
                # Default to popular vehicle types
                vehicle_styles = ["SUV", "Sedan", "Crossover"]

            logger.info(f"Emergency suggestions targeting vehicle styles: {vehicle_styles}")

            # Get more cars from search based on the query with targeted vehicle styles
            search_params = {
                "page": 1,
                "page_size": 20,  # Increased to get more options
            }

            # Try to match vehicle style if possible
            if vehicle_styles:
                search_params["vehicle_style"] = vehicle_styles[0]  # Use first style as primary

            # Also include any brand mentioned in the query
            car_brands = ["Toyota", "Honda", "Ford", "Chevrolet", "BMW", "Mercedes",
                          "Audi", "Lexus", "Nissan", "Hyundai", "Kia", "Subaru",
                          "Volkswagen", "Mazda", "Jeep", "Tesla", "Volvo"]

            for brand in car_brands:
                if brand.lower() in query_lower:
                    search_params["make"] = brand
                    break

            # Get cars
            search_results = await self.car_service.search_cars_from_csv(**search_params)
            cars = search_results.get("items", [])

            # If we don't have enough cars, try with a broader search
            if len(cars) < min_needed and vehicle_styles and len(vehicle_styles) > 1:
                # Try another vehicle style from our list
                backup_search_params = search_params.copy()
                backup_search_params["vehicle_style"] = vehicle_styles[1]
                if "make" in backup_search_params:
                    # Remove brand restriction for more results
                    del backup_search_params["make"]

                backup_results = await self.car_service.search_cars_from_csv(**backup_search_params)
                backup_cars = backup_results.get("items", [])

                # Extend our car list
                cars.extend(backup_cars)

            # If we still don't have cars, try popular family vehicles as fallback
            if len(cars) < min_needed and "suv" in query_lower:
                # Hardcoded popular family SUVs as last resort
                popular_suvs = [
                    {"name": "Toyota RAV4", "brand": "Toyota", "model": "RAV4"},
                    {"name": "Honda CR-V", "brand": "Honda", "model": "CR-V"},
                    {"name": "Toyota Highlander", "brand": "Toyota", "model": "Highlander"},
                    {"name": "Honda Pilot", "brand": "Honda", "model": "Pilot"},
                    {"name": "Ford Explorer", "brand": "Ford", "model": "Explorer"},
                    {"name": "Chevrolet Traverse", "brand": "Chevrolet", "model": "Traverse"}
                ]

                for suv in popular_suvs:
                    # Try to find this car in the database
                    car_data = await self._get_car_data_by_name(suv["brand"], suv["model"])
                    if car_data and "id" in car_data:
                        cars.append(car_data)

                logger.info(f"Added popular family SUVs as fallback options")

            # Create a set to track unique brand-model combinations
            seen_brand_models = set()

            # Create suggestions from search results with diversity
            # Track brands and models to ensure diversity
            selected_brands = set()
            selected_models = set()

            # First pass: try to get diverse options by brand and model
            for car in cars:
                try:
                    # Get car id from the 'id' field
                    car_id = car.get("id")

                    # Skip if invalid ID
                    if car_id is None:
                        logger.warning(f"Car has no ID, skipping: {car.get('brand', '')} {car.get('model', '')}")
                        continue

                    # Verify this car exists in the database
                    car_db_data = await self._get_car_data(car_id)
                    if not car_db_data or 'id' not in car_db_data:
                        logger.warning(f"Car ID {car_id} not verified in database")
                        continue

                    # Use the verified database ID and data
                    verified_car_id = car_db_data['id']

                    brand = car_db_data.get('brand', '').strip()
                    model = car_db_data.get('model', '').strip()

                    if not brand or not model:
                        continue

                    # Check for duplicate brand-model
                    brand_model_key = f"{brand.lower()}_{model.lower()}"
                    if brand_model_key in seen_brand_models:
                        continue

                    seen_brand_models.add(brand_model_key)

                    # Skip if we already have this brand-model combination
                    if brand_model_key in selected_models:
                        continue

                    # Limit the number of cars from the same brand to ensure diversity
                    if brand in selected_brands and len(selected_brands) >= 2:
                        continue

                    # Verify the vehicle style matches if this is a specific query
                    car_style = car_db_data.get('vehicle_style', '')
                    if vehicle_styles and car_style and not any(style.lower() in car_style.lower() for style in vehicle_styles):
                        logger.info(f"Skipping {brand} {model} - style {car_style} doesn't match query requirements {vehicle_styles}")
                        continue

                    car_suggestion = CarSuggestion(
                        id=str(verified_car_id),  # Use verified id
                        name=f"{brand} {model}",
                        brand=brand,
                        model=model,
                        price=float(car_db_data.get('msrp', 0)),
                        image_url=car_db_data.get('image_url', None),
                        reasons=[f"Recommended {car_db_data.get('vehicle_style', 'vehicle')}"]
                    )

                    # Convert to dict using model_dump
                    suggestion_dict = car_suggestion.model_dump()

                    suggestions.append(suggestion_dict)
                    selected_brands.add(brand)
                    selected_models.add(brand_model_key)
                    logger.info(f"Added emergency suggestion: {brand} {model} (ID: {verified_car_id})")

                    # Stop once we have enough diverse suggestions
                    if len(suggestions) >= min_needed:
                        break

                except Exception as e:
                    logger.error(f"Error creating emergency suggestion: {e}")
                    continue

            # If we don't have enough suggestions, add more from the remaining cars
            # For the second pass, we'll be less strict about brand diversity
            if len(suggestions) < min_needed:
                logger.info(f"Not enough diverse suggestions ({len(suggestions)}), adding more")

                # Reset the seen brands for the second pass to allow more from the same brand
                # but keep track of exact models we've already added
                for car in cars:
                    if len(suggestions) >= min_needed:
                        break

                    try:
                        # Get car id from the 'id' field
                        car_id = car.get("id")

                        # Skip if invalid ID
                        if car_id is None:
                            continue

                        # Verify this car exists in the database
                        car_db_data = await self._get_car_data(car_id)
                        if not car_db_data or 'id' not in car_db_data:
                            continue

                        # Use the verified database ID and data
                        verified_car_id = car_db_data['id']

                        brand = car_db_data.get('brand', '').strip()
                        model = car_db_data.get('model', '').strip()

                        if not brand or not model:
                            continue

                        # Check for duplicate brand-model
                        brand_model_key = f"{brand.lower()}_{model.lower()}"
                        if brand_model_key in seen_brand_models:
                            continue

                        seen_brand_models.add(brand_model_key)

                        car_suggestion = CarSuggestion(
                            id=str(verified_car_id),  # Use verified id
                            name=f"{brand} {model}",
                            brand=brand,
                            model=model,
                            price=float(car_db_data.get('msrp', 0)),
                            image_url=car_db_data.get('image_url', None),
                            reasons=[f"Recommended {car_db_data.get('vehicle_style', 'vehicle')}"]
                        )

                        # Convert to dict using model_dump
                        suggestion_dict = car_suggestion.model_dump()

                        suggestions.append(suggestion_dict)
                        logger.info(f"Added additional emergency suggestion: {brand} {model} (ID: {verified_car_id})")
                    except Exception as e:
                        logger.error(f"Error creating additional emergency suggestion: {e}")
                        continue

            logger.info(f"Created {len(suggestions)} emergency suggestions")
            return suggestions

        except Exception as e:
            logger.error(f"Error getting emergency suggestions: {e}")
            return []

class LlamaIndexVectorAdapter:
    """
    Adapter đơn giản để đa dạng hóa kết quả vector search
    """
    def __init__(self, vector_db):
        self.vector_db = vector_db

    async def query(self, query_str, top_k=5, filters=None):
        """
        Thực hiện truy vấn vector search với đa dạng hóa kết quả

        Args:
            query_str: Câu truy vấn
            top_k: Số lượng kết quả trả về
            filters: Bộ lọc metadata (nếu có)

        Returns:
            Danh sách các tài liệu phù hợp
        """
        try:
            # Sử dụng vector_db.similarity_search để lấy kết quả
            results = await self.vector_db.similarity_search(query_str, k=top_k*3)

            if not results:
                return []

            # Áp dụng bộ lọc nếu có
            filtered_results = []
            if filters:
                # Lọc kết quả dựa trên metadata
                for doc in results:
                    if hasattr(doc, 'metadata') and self._apply_filters(doc.metadata, filters):
                        filtered_results.append(doc)

                # Nếu không có kết quả sau khi lọc, sử dụng kết quả gốc
                if not filtered_results:
                    filtered_results = results
            else:
                filtered_results = results

            # Đa dạng hóa kết quả
            diverse_results = self._diversify_results(filtered_results, top_k)

            return diverse_results[:top_k]

        except Exception as e:
            logger.error(f"Error in simple vector adapter query: {e}")
            # Fallback sang cách cũ
            return await self.vector_db.similarity_search(query_str, k=top_k)

    def _apply_filters(self, metadata, filters):
        """Áp dụng bộ lọc vào metadata"""
        if not filters or not metadata:
            return True

        # Kiểm tra từng trường với bộ lọc
        for field, value in filters.items():
            if field not in metadata:
                return False

            if metadata[field] != value:
                return False

        return True

    def _diversify_results(self, docs, top_k):
        """Đa dạng hóa kết quả dựa trên thương hiệu, loại xe"""
        if not docs:
            return []

        import random

        # Theo dõi thương hiệu và loại xe để đảm bảo đa dạng
        brands = set()
        types = set()
        diverse_docs = []

        # Ưu tiên các document có điểm cao nhất trước
        for doc in docs:
            if not hasattr(doc, 'metadata'):
                continue

            # Lấy thông tin thương hiệu và loại xe từ metadata
            metadata = doc.metadata
            brand = metadata.get('brand', None)
            car_type = metadata.get('vehicle_type', metadata.get('vehicle_style', None))

            # Giới hạn số lượng document từ cùng một thương hiệu/loại xe
            if brand and brand in brands and len(brands) >= 3:
                continue

            if car_type and car_type in types and len(types) >= 3:
                continue

            # Thêm document vào kết quả
            diverse_docs.append(doc)
            if brand:
                brands.add(brand)
            if car_type:
                types.add(car_type)

            # Dừng khi đã có đủ kết quả
            if len(diverse_docs) >= top_k:
                break

        # Nếu chưa đủ kết quả, thêm các document còn lại
        if len(diverse_docs) < top_k:
            remaining = [d for d in docs if d not in diverse_docs]
            random.shuffle(remaining)  # Xáo trộn kết quả để tăng tính đa dạng
            diverse_docs.extend(remaining[:top_k - len(diverse_docs)])

        return diverse_docs

rag_service = RAGService()