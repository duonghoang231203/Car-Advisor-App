from typing import Dict, List, Any
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableSequence
from app.db.vector_store import vector_db
from app.config import settings
from app.models.car import Car
from app.models.chat import CarSuggestion
from app.services.car_service import CarService
import json
from app.core.logging import logger
import re

class RAGService:
    def __init__(self):
        """Initialize the RAG service"""
        self.initialize_llm()
        self.qa_chain = None
        self.initialize_qa_chain()
        self.car_service = CarService()

    def initialize_llm(self):
        """Initialize the language model"""
        try:
            if settings.LLM_TYPE == "openai":
                self.llm = ChatOpenAI(
                    model_name=settings.LLM_MODEL,
                    openai_api_key=settings.LLM_API_KEY,
                    temperature=0.7
                )

                # Create a more creative version of the LLM with higher temperature
                # for generating more varied responses
                self.creative_llm = ChatOpenAI(
                    model_name=settings.LLM_MODEL,
                    openai_api_key=settings.LLM_API_KEY,
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

    async def process_query(self, query, conversation_history=None, k=10):
        """
        Process a user query and return a response with car suggestions

        Args:
            query: The current user query
            conversation_history: List of previous messages in the conversation
            k: Number of documents to retrieve
        """
        try:
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

            # Check if this is a follow-up question
            is_follow_up = await self._is_follow_up_question(query, conversation_history)
            logger.info(f"Query: '{query}' - Is follow-up: {is_follow_up}")

            # Adjust search parameters based on whether this is a follow-up
            search_k = k
            if is_follow_up:
                # For follow-up questions, we might want to retrieve more documents
                # to ensure we have enough context to answer the follow-up
                search_k = k + 5

            # Step 1: Get relevant car documents
            docs = await vector_db.similarity_search(query, k=search_k)
            context = "\n".join([doc.page_content for doc in docs]) if docs else ""

            # Step 2: Extract relevant cars for suggestions with diversity
            suggestions = []
            selected_brands = set()
            selected_models = set()
            min_suggestions = 3  # Always return at least 3 car suggestions
            max_suggestions = 6  # Maximum number of suggestions

            if docs:
                # First pass: process all docs to get car data
                car_data_list = []
                for doc in docs:  # Process all docs, not just top 3
                    if doc.metadata and "car_id" in doc.metadata:
                        car_id = doc.metadata["car_id"]
                        car_data = await self._get_car_data(car_id)
                        if car_data:
                            car_data_list.append((car_id, car_data))

                # Second pass: select diverse cars
                for car_id, car_data in car_data_list:
                    try:
                        brand = car_data.get('brand', '')
                        model = car_data.get('model', '')

                        # Skip if we already have this brand-model combination
                        brand_model_key = f"{brand}_{model}"
                        if brand_model_key in selected_models:
                            continue

                        # Limit the number of cars from the same brand to ensure diversity
                        if brand in selected_brands and len(selected_brands) >= 2:
                            continue

                        car_suggestion = CarSuggestion(
                            car_id=str(car_id),
                            name=f"{brand} {model}",
                            brand=brand,
                            model=model,
                            price=float(car_data.get('msrp', 0)),
                            image_url=car_data.get('image_url', None),
                            reasons=[]  # Will be populated by explanation
                        )

                        # Convert to dict using model_dump for Pydantic v2 or dict for v1
                        if hasattr(car_suggestion, 'model_dump'):
                            suggestion_dict = car_suggestion.model_dump()
                        else:
                            suggestion_dict = car_suggestion.dict()

                        suggestions.append(suggestion_dict)
                        selected_brands.add(brand)
                        selected_models.add(brand_model_key)

                        # Stop once we have enough diverse suggestions
                        if len(suggestions) >= max_suggestions:
                            break
                    except Exception as e:
                        logger.error(f"Error creating car suggestion: {e}")

            # Step 3: If no suggestions or too few suggestions were found via vector search,
            # try direct category match to ensure we have at least 3 options
            if len(suggestions) < min_suggestions:
                logger.info(f"Only found {len(suggestions)} suggestions from vector search, using fallback mechanism to get at least {min_suggestions}")
                fallback_suggestions = await self._get_fallback_suggestions(query)

                # Add fallback suggestions without duplicating cars
                existing_car_ids = {s.get('car_id') for s in suggestions}
                for suggestion in fallback_suggestions:
                    if suggestion.get('car_id') not in existing_car_ids and len(suggestions) < max_suggestions:
                        suggestions.append(suggestion)
                        existing_car_ids.add(suggestion.get('car_id'))

                # If we got fallback suggestions, update the context with information about these cars
                if suggestions:
                    car_desc = []
                    for car in suggestions:
                        car_desc.append(f"{car.get('brand', '')} {car.get('model', '')} (MSRP: ${car.get('price', 0):,.2f})")

                    additional_context = f"\nRelevant cars for this query include: {', '.join(car_desc)}.\n"
                    context = context + additional_context if context else additional_context

            # Step 4: Generate an explanation based on the suggestions and conversation history
            explanation = await self._generate_explanation(
                suggestions=suggestions,
                query=query,
                conversation_history=conversation_history
            )

            # Step 5: Now generate the answer using the QA chain with both the document context and suggestion info
            suggestion_context = ""
            if suggestions:
                car_names = [f"{s.get('brand', '')} {s.get('model', '')}" for s in suggestions]
                suggestion_context = f"\nBased on the query, the following cars could be recommended: {', '.join(car_names)}.\n{explanation}\n"

            # Combine original context with suggestion context
            enhanced_context = f"{context}\n{suggestion_context}".strip()

            # Choose a response style based on conversation context and randomness for variety
            import random

            # For follow-up questions, we'll use different approaches for more varied responses
            if is_follow_up:
                # Randomly select between different response styles for follow-ups
                # This adds variety and prevents the chatbot from sounding repetitive
                response_style = random.choice(["enthusiastic", "concise", "custom"])

                if response_style == "enthusiastic" and self.enthusiastic_qa_chain:
                    # Use the enthusiastic QA chain
                    logger.info("Using enthusiastic response style for follow-up")
                    response_obj = await self.enthusiastic_qa_chain.ainvoke({
                        "context": enhanced_context,
                        "query": query,
                        "conversation_history": formatted_history,
                        "previously_mentioned_cars": ', '.join(previously_mentioned_cars) if previously_mentioned_cars else 'None'
                    })
                    response = response_obj.content if hasattr(response_obj, 'content') else response_obj

                elif response_style == "concise" and self.concise_qa_chain:
                    # Use the concise QA chain
                    logger.info("Using concise response style for follow-up")
                    response_obj = await self.concise_qa_chain.ainvoke({
                        "context": enhanced_context,
                        "query": query,
                        "conversation_history": formatted_history,
                        "previously_mentioned_cars": ', '.join(previously_mentioned_cars) if previously_mentioned_cars else 'None'
                    })
                    response = response_obj.content if hasattr(response_obj, 'content') else response_obj

                else:
                    # Use a custom prompt for follow-up questions
                    logger.info("Using custom prompt for follow-up")
                    follow_up_prompt = f"""
                    You are a car advisor helping customers choose cars.

                    Context information: {enhanced_context}

                    Conversation history:
                    {formatted_history}

                    Current question: {query}

                    Previously mentioned cars: {', '.join(previously_mentioned_cars) if previously_mentioned_cars else 'None'}

                    IMPORTANT INSTRUCTIONS:
                    1. ALWAYS discuss MULTIPLE car options (at least 3 different cars) in your responses.
                    2. If this is an initial question, present and compare at least 3 different car options.
                    3. You MUST refer to the same cars mentioned in previous messages for follow-up questions.
                    4. Do NOT introduce new cars unless specifically asked for alternatives.
                    5. Maintain continuity with previous responses - if you recommended specific cars before, continue discussing those same cars.
                    6. Be specific and reference details from the conversation history.
                    7. If you're asked about features, specifications, or comparisons, refer to the exact cars mentioned earlier.
                    8. NEVER recommend just one car - always provide multiple options for comparison.

                    Provide a helpful, conversational response that directly addresses the question.
                    Make sure to maintain continuity with the previous conversation.
                    Be specific and reference details from the conversation history when relevant.
                    Vary your response style to sound natural and engaging.
                    """

                    # Use the creative LLM for custom follow-up questions
                    response_obj = await self.creative_llm.ainvoke(follow_up_prompt)
                    response = response_obj.content
            else:
                # For initial questions, use the standard QA chain most of the time
                # but occasionally use other styles for variety
                response_style = random.choices(
                    ["standard", "enthusiastic", "concise"],
                    weights=[0.7, 0.15, 0.15],
                    k=1
                )[0]

                if response_style == "enthusiastic" and self.enthusiastic_qa_chain:
                    logger.info("Using enthusiastic response style for initial question")
                    response_obj = await self.enthusiastic_qa_chain.ainvoke({
                        "context": enhanced_context,
                        "query": query,
                        "conversation_history": formatted_history,
                        "previously_mentioned_cars": ', '.join(previously_mentioned_cars) if previously_mentioned_cars else 'None'
                    })
                    response = response_obj.content if hasattr(response_obj, 'content') else response_obj

                elif response_style == "concise" and self.concise_qa_chain:
                    logger.info("Using concise response style for initial question")
                    response_obj = await self.concise_qa_chain.ainvoke({
                        "context": enhanced_context,
                        "query": query,
                        "conversation_history": formatted_history,
                        "previously_mentioned_cars": ', '.join(previously_mentioned_cars) if previously_mentioned_cars else 'None'
                    })
                    response = response_obj.content if hasattr(response_obj, 'content') else response_obj

                else:
                    # Use the standard QA chain for most initial questions
                    logger.info("Using standard response style")
                    response_obj = await self.qa_chain.ainvoke({
                        "context": enhanced_context,
                        "query": query,
                        "conversation_history": formatted_history,
                        "previously_mentioned_cars": ', '.join(previously_mentioned_cars) if previously_mentioned_cars else 'None'
                    })
                    response = response_obj.content if hasattr(response_obj, 'content') else response_obj

            logger.info(f"Query: {query}")
            logger.info(f"Found {len(suggestions)} suggestions")

            # Get the response text
            response_text = response if isinstance(response, str) else response.content

            # Extract car mentions from the response text to ensure suggestions match
            # This is a critical step to ensure the suggestions match what's mentioned in the text
            extracted_cars = self._extract_cars_from_text(response_text)
            logger.info(f"Extracted cars from response: {extracted_cars}")

            # If we found cars in the response text, update suggestions to match
            if extracted_cars and len(extracted_cars) >= 3:
                # Create new suggestions based on extracted cars
                updated_suggestions = []
                used_names = set()  # Track used names to ensure uniqueness

                for car in extracted_cars:
                    # Skip if we already have this exact car name
                    if car['name'] in used_names:
                        continue

                    # Get car data for this car
                    car_data = await self._get_car_data_by_name(car['brand'], car['model'])

                    if car_data:
                        car_id = car_data.get('id', '0')
                        car_suggestion = CarSuggestion(
                            car_id=str(car_id),
                            name=car['name'],
                            brand=car['brand'],
                            model=car['model'],
                            price=float(car_data.get('msrp', car.get('price', 0))),
                            image_url=car_data.get('image_url', None),
                            reasons=[f"Recommended {car_data.get('vehicle_style', 'vehicle')}"]
                        )

                        # Convert to dict
                        if hasattr(car_suggestion, 'model_dump'):
                            suggestion_dict = car_suggestion.model_dump()
                        else:
                            suggestion_dict = car_suggestion.dict()

                        updated_suggestions.append(suggestion_dict)
                        used_names.add(car['name'])

                        # Stop once we have 3 unique car suggestions
                        if len(updated_suggestions) >= 3:
                            break

                # If we found at least 3 cars, use the updated suggestions
                if len(updated_suggestions) >= 3:
                    suggestions = updated_suggestions
                    logger.info(f"Updated suggestions to match response text: {[s.get('name') for s in suggestions]}")

            return {
                "response": response_text,
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

    async def _is_follow_up_question(self, query, conversation_history):
        """
        Determine if the current query is a follow-up to previous conversation

        Args:
            query: Current user query
            conversation_history: List of previous messages

        Returns:
            bool: True if this is a follow-up question, False otherwise
        """
        if not conversation_history or len(conversation_history) < 2:
            return False

        # Check for pronouns and references to previous content
        follow_up_indicators = [
            "it", "they", "them", "those", "these", "that", "this",
            "the car", "the vehicle", "the option", "the model",
            "what about", "how about", "tell me more", "more details",
            "which one", "which of", "between these", "of these",
            "yes", "no", "why", "how", "when", "where", "who"
        ]

        query_lower = query.lower()

        # Check if query starts with a follow-up indicator
        for indicator in follow_up_indicators:
            if query_lower.startswith(indicator) or f" {indicator} " in query_lower:
                return True

        # Check if query is very short (likely a follow-up)
        if len(query.split()) <= 3:
            return True

        return False

    async def _get_car_data(self, car_id):
        """Get car data by ID"""
        try:
            # Pass the car_id directly to get_car_by_id, which now handles string IDs
            car_data = await self.car_service.get_car_by_id(car_id)
            return car_data
        except Exception as e:
            logger.error(f"Error getting car data for ID {car_id}: {e}")
            return None

    async def _get_fallback_suggestions(self, query, response_text=None):
        """Get fallback suggestions based on query content"""
        suggestions = []
        try:
            # Extract car type from query and response (if provided)
            car_types = {
                "sedan": ["sedan", "4-door", "family car"],
                "suv": ["suv", "crossover", "sport utility vehicle"],
                "truck": ["truck", "pickup", "utility vehicle"],
                "hybrid": ["hybrid", "electric", "eco-friendly"],
                "luxury": ["luxury", "premium", "high-end"],
                "sports": ["sports", "performance", "fast"]
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

            # If no type was detected, default to sedan
            if not detected_type:
                detected_type = "sedan"
                logger.info(f"No car type detected in query, defaulting to {detected_type}")

            # Get more cars from search based on detected type to ensure diversity
            search_params = {
                "vehicle_style": detected_type if detected_type != "luxury" and detected_type != "hybrid" else None,
                "page": 1,
                "page_size": 15  # Increased from 6 to 15 to get more options
            }

            # Add market category search for luxury and hybrid
            if detected_type == "luxury" or detected_type == "hybrid":
                search_params["search_query"] = detected_type

            # Get cars
            search_results = await self.car_service.search_cars_from_csv(**search_params)
            cars = search_results.get("items", [])

            # Create suggestions from search results with diversity
            # Track brands and models to ensure diversity
            selected_brands = set()
            selected_models = set()
            max_suggestions = 6  # Increased from 3 to 6

            # First pass: try to get diverse options by brand and model
            for car in cars:
                try:
                    brand = car.get('brand', '')
                    model = car.get('model', '')

                    # Skip if we already have this brand-model combination
                    brand_model_key = f"{brand}_{model}"
                    if brand_model_key in selected_models:
                        continue

                    # Limit the number of cars from the same brand to ensure diversity
                    if brand in selected_brands and len(selected_brands) >= 2:
                        continue

                    car_id = car.get("id", 0)  # Use position as fallback ID
                    car_suggestion = CarSuggestion(
                        car_id=str(car_id),
                        name=f"{brand} {model}",
                        brand=brand,
                        model=model,
                        price=float(car.get('msrp', 0)),
                        image_url=car.get('image_url', None),
                        reasons=[f"Recommended {detected_type.capitalize()} vehicle"]
                    )

                    # Convert to dict
                    if hasattr(car_suggestion, 'model_dump'):
                        suggestion_dict = car_suggestion.model_dump()
                    else:
                        suggestion_dict = car_suggestion.dict()

                    suggestions.append(suggestion_dict)
                    selected_brands.add(brand)
                    selected_models.add(brand_model_key)

                    # Stop once we have enough diverse suggestions
                    if len(suggestions) >= max_suggestions:
                        break

                except Exception as e:
                    logger.error(f"Error creating fallback suggestion: {e}")
                    continue

            # If we don't have enough suggestions, add more from the remaining cars
            if len(suggestions) < 3:
                logger.info(f"Not enough diverse suggestions ({len(suggestions)}), adding more")
                for car in cars:
                    if len(suggestions) >= 3:
                        break

                    try:
                        car_id = car.get("id", 0)

                        # Skip if this car is already in suggestions
                        if any(s.get('car_id') == str(car_id) for s in suggestions):
                            continue

                        brand = car.get('brand', '')
                        model = car.get('model', '')

                        car_suggestion = CarSuggestion(
                            car_id=str(car_id),
                            name=f"{brand} {model}",
                            brand=brand,
                            model=model,
                            price=float(car.get('msrp', 0)),
                            image_url=car.get('image_url', None),
                            reasons=[f"Recommended {detected_type.capitalize()} vehicle"]
                        )

                        # Convert to dict
                        if hasattr(car_suggestion, 'model_dump'):
                            suggestion_dict = car_suggestion.model_dump()
                        else:
                            suggestion_dict = car_suggestion.dict()

                        suggestions.append(suggestion_dict)
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
                         "Volkswagen", "Mazda", "Jeep", "Tesla", "Volvo"]

            extracted_cars = []

            for brand in car_brands:
                # Look for brand followed by model
                matches = re.findall(f"{brand}\\s+([\\w-]+)", text)
                for model in matches:
                    car = {
                        'brand': brand,
                        'model': model,
                        'name': f"{brand} {model}",
                        'price': 0  # Default price
                    }

                    # Check if this car is already in the list
                    if not any(c['name'] == car['name'] for c in extracted_cars):
                        extracted_cars.append(car)

            return extracted_cars
        except Exception as e:
            logger.error(f"Error extracting cars from text: {e}")
            return []

    async def _get_car_data_by_name(self, brand, model):
        """
        Get car data by brand and model

        Args:
            brand: Car brand
            model: Car model

        Returns:
            Car data dictionary or None if not found
        """
        try:
            # Search for cars with this brand and model
            search_params = {
                "make": brand,
                "model": model,
                "page": 1,
                "page_size": 1
            }

            # Get cars
            search_results = await self.car_service.search_cars_from_csv(**search_params)
            cars = search_results.get("items", [])

            if cars and len(cars) > 0:
                return cars[0]

            # If not found, try a more general search with just the brand
            search_params = {
                "make": brand,
                "page": 1,
                "page_size": 10
            }

            search_results = await self.car_service.search_cars_from_csv(**search_params)
            cars = search_results.get("items", [])

            # Try to find a car with the matching model
            for car in cars:
                if car.get('model', '').lower() == model.lower():
                    return car

            # If still not found, return the first car of this brand
            if cars and len(cars) > 0:
                return cars[0]

            # If all else fails, create a default car
            return {
                "id": "0",
                "name": f"{brand} {model}",
                "brand": brand,
                "model": model,
                "year": 2023,
                "price": 35000.0,
                "msrp": 35000.0,
                "vehicle_style": "SUV" if "SUV" in model.upper() else "Sedan"
            }

        except Exception as e:
            logger.error(f"Error getting car data by name: {e}")
            # Return a default car
            return {
                "id": "0",
                "name": f"{brand} {model}",
                "brand": brand,
                "model": model,
                "year": 2023,
                "price": 35000.0,
                "msrp": 35000.0,
                "vehicle_style": "SUV" if "SUV" in model.upper() else "Sedan"
            }

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

rag_service = RAGService()