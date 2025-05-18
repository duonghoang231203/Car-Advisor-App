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
from pydantic import BaseModel, Field
from typing import Optional, List as TypedList

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
        logger.info("RAGService initialized with simplified vector adapter")

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

            # Enhance the query to make vector search more effective
            enhanced_query = await self._enhance_query(query, conversation_history)
            logger.info(f"Enhanced query: '{enhanced_query}'")

            # Adjust search parameters based on whether this is a follow-up
            search_k = k
            if is_follow_up:
                # For follow-up questions, we might want to retrieve more documents
                # to ensure we have enough context to answer the follow-up
                search_k = k + 5

            # Tạo bộ lọc nếu cần (ví dụ lọc theo loại xe được đề cập trong truy vấn)
            filters = self._create_filters_from_query(enhanced_query)
                
            # Step 1: Get relevant car documents using LlamaIndex adapter
            docs = await self.vector_adapter.query(enhanced_query, top_k=search_k, filters=filters)
            context = "\n".join([doc.page_content for doc in docs]) if docs else ""

            # Step 2: Generate a preliminary response
            # Step 2a: Generate initial response using llama-index first (if available)
            initial_response_text = await self._generate_response_with_llama_index(
                query=query,
                docs=docs,
                conversation_history=conversation_history
            )
            
            if not initial_response_text:
                # Fallback to the old method
                suggestion_context = "Based on your query, I'll recommend some suitable cars for your needs."
                
                # Combine original context with suggestion context
                enhanced_context = f"{context}\n{suggestion_context}".strip()
                
                # Choose a response style based on conversation context
                import random
                response_style = random.choices(
                    ["standard", "enthusiastic", "concise"],
                    weights=[0.7, 0.15, 0.15],
                    k=1
                )[0]
                
                if response_style == "enthusiastic" and self.enthusiastic_qa_chain:
                    logger.info("Using enthusiastic response style")
                    response_obj = await self.enthusiastic_qa_chain.ainvoke({
                        "context": enhanced_context,
                        "query": query,
                        "conversation_history": formatted_history,
                        "previously_mentioned_cars": ', '.join(previously_mentioned_cars) if previously_mentioned_cars else 'None'
                    })
                    response = response_obj.content if hasattr(response_obj, 'content') else response_obj
                    
                elif response_style == "concise" and self.concise_qa_chain:
                    logger.info("Using concise response style")
                    response_obj = await self.concise_qa_chain.ainvoke({
                        "context": enhanced_context,
                        "query": query,
                        "conversation_history": formatted_history,
                        "previously_mentioned_cars": ', '.join(previously_mentioned_cars) if previously_mentioned_cars else 'None'
                    })
                    response = response_obj.content if hasattr(response_obj, 'content') else response_obj
                    
                else:
                    # Use the standard QA chain
                    logger.info("Using standard response style")
                    response_obj = await self.qa_chain.ainvoke({
                        "context": enhanced_context,
                        "query": query,
                        "conversation_history": formatted_history,
                        "previously_mentioned_cars": ', '.join(previously_mentioned_cars) if previously_mentioned_cars else 'None'
                    })
                    response = response_obj.content if hasattr(response_obj, 'content') else response_obj
                    
                # Get the response text
                initial_response_text = response if isinstance(response, str) else response.content

            # Step 3: Generate an explanation that can be parsed for car suggestions
            # This is now moved earlier in the process
            extended_prompt = f"""
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
                response_obj = await self.creative_llm.ainvoke(extended_prompt)
                explanation = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
            else:
                response_obj = await self.llm.ainvoke(extended_prompt)
                explanation = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)

            logger.info(f"Generated explanation with specific car mentions")
            
            # Step 4: Extract car mentions from the explanation
            extracted_cars = self._extract_cars_from_text(explanation)
            logger.info(f"Extracted cars from explanation: {extracted_cars}")
            
            # Step 5: Look up these cars in the database and build suggestions
            suggestions = []
            used_names = set()  # Track used names to ensure uniqueness
            
            # Process each extracted car
            for car in extracted_cars:
                # Skip if we already have this exact car name
                if car['name'] in used_names:
                    continue

                # Get car data for this car
                car_data = await self._get_car_data_by_name(car['brand'], car['model'])

                if car_data and 'id' in car_data and car_data['id'] is not None:
                    # Get car_id from the 'id' field
                    car_id = car_data.get('id')
                    
                    # Double-check that this car exists in the database
                    car_db_data = await self._get_car_data(car_id)
                    
                    if car_db_data and 'id' in car_db_data:
                        # We have a confirmed database entry
                        logger.info(f"Found valid database entry for {car['name']}: ID {car_id}")
                        
                        car_suggestion = CarSuggestion(
                            id=str(car_id),  # Convert ID to string for consistency
                            name=f"{car_db_data.get('brand', car['brand'])} {car_db_data.get('model', car['model'])}",
                            brand=car_db_data.get('brand', car['brand']),
                            model=car_db_data.get('model', car['model']),
                            price=float(car_db_data.get('msrp', car_data.get('price', 0))),
                            image_url=car_db_data.get('image_url', None),
                            reasons=[f"Recommended {car_db_data.get('vehicle_style', 'vehicle')}"]
                        )

                        # Convert to dict using model_dump
                        suggestion_dict = car_suggestion.model_dump()

                        suggestions.append(suggestion_dict)
                        used_names.add(car['name'])
                    else:
                        logger.warning(f"Car with ID {car_id} not found in database verification")
                else:
                    logger.warning(f"No valid car data found for {car['name']}")
            
            # Check if we have enough suggestions (at least 3)
            if len(suggestions) < 3:
                logger.warning(f"Not enough suggestions found from explanation ({len(suggestions)}), adding fallback suggestions")
                fallback_suggestions = await self._get_fallback_suggestions(query, explanation)
                
                # Add fallback suggestions without duplicating cars
                existing_car_ids = set(s.get('id') for s in suggestions if s.get('id'))
                for suggestion in fallback_suggestions:
                    car_id = suggestion.get('id')
                    if car_id and car_id not in existing_car_ids:
                        suggestions.append(suggestion)
                        existing_car_ids.add(car_id)

            # Ensure we have a good diversity of brands and models
            suggestions = await self._ensure_suggestion_diversity(suggestions)
            
            # Log the final diverse suggestions that will be used
            logger.info(f"Final diverse suggestions: {[f'{s.get('brand', '')} {s.get('model', '')}' for s in suggestions]}")
            
            # FINAL STEP: Make one more LLM call to combine suggestions and explanation into a final response
            # This ensures the response specifically addresses the suggestions we're going to return
            final_response_text, updated_suggestions = await self._generate_final_response(
                initial_response=initial_response_text,
                suggestions=suggestions,
                explanation=explanation,
                query=query,
                conversation_history=conversation_history
            )
            
            # Use updated suggestions if they were modified during final response generation
            if updated_suggestions:
                suggestions = updated_suggestions
                logger.info(f"Using updated suggestions: {[f'{s.get('brand', '')} {s.get('model', '')}' for s in suggestions]}")

            # FINAL VALIDATION: Ensure we have at least 3 distinct car options
            # This is a critical requirement that must be met
            if len(suggestions) < 3:
                logger.warning(f"CRITICAL: Final suggestions count ({len(suggestions)}) is less than required minimum of 3")
                
                # Get emergency fallback suggestions from a broader search
                emergency_suggestions = await self._get_emergency_suggestions(query, min_needed=3-len(suggestions))
                
                # Add any new suggestions we found
                existing_ids = set(s.get('id') for s in suggestions if s.get('id'))
                for suggestion in emergency_suggestions:
                    if suggestion.get('id') not in existing_ids:
                        suggestions.append(suggestion)
                        existing_ids.add(suggestion.get('id'))
                        logger.info(f"Added emergency suggestion: {suggestion.get('brand', '')} {suggestion.get('model', '')}")
                
                # Final check - if we still don't have 3, log a critical error
                if len(suggestions) < 3:
                    logger.error(f"CRITICAL FAILURE: Unable to generate 3 car suggestions, only have {len(suggestions)}")

            # One last check to ensure all suggestions have required fields
            for suggestion in suggestions:
                if not suggestion.get('id') or not suggestion.get('brand') or not suggestion.get('model'):
                    logger.warning(f"Invalid suggestion detected: {suggestion}")

            return {
                "response": final_response_text,
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
            # Convert car_id to integer if it's a string to ensure proper database lookup
            if isinstance(car_id, str):
                try:
                    car_id = int(car_id)
                except ValueError:
                    logger.error(f"Invalid car ID format: {car_id}, cannot convert to int")
                    return None

            # Pass the car_id to get_car_by_id
            car_data = await self.car_service.get_car_by_id(car_id)
            
            # If no car was found, return None
            if not car_data:
                logger.warning(f"No car found with ID {car_id}")
                return None

            # Ensure the returned car_data has an 'id' field that matches the database ID
            if 'id' not in car_data:
                logger.warning(f"Car data missing ID field, adding ID {car_id}")
                car_data['id'] = car_id
            elif car_data['id'] != car_id:
                logger.warning(f"Car ID mismatch: requested {car_id}, received {car_data['id']}")
                # Keep the database ID as it's more authoritative

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

            # Create a set to track unique brand-model combinations
            seen_brand_models = set()

            # Create suggestions from search results with diversity
            # Track brands and models to ensure diversity
            selected_brands = set()
            selected_models = set()
            max_suggestions = 6  # Increased from 3 to 6

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
        Get car data by brand and model

        Args:
            brand: Car brand
            model: Car model

        Returns:
            Car data dictionary or None if not found
        """
        try:
            # Log the search request
            logger.info(f"Searching for car: {brand} {model}")
            
            # First search for exact match with both brand and model
            search_params = {
                "make": brand,
                "model": model,
                "page": 1,
                "page_size": 10,  # Get more results to find the best match
                "partial_match": False  # Use exact matching first
            }

            # Get cars
            search_results = await self.car_service.search_cars_from_csv(**search_params)
            cars = search_results.get("items", [])
            
            # If we found results, find the best match
            if cars and len(cars) > 0:
                # Try to find exact match by model
                for car in cars:
                    # Check for exact match on model (case insensitive)
                    db_model = car.get('model', '')
                    if db_model.lower() == model.lower():
                        logger.info(f"Found exact match: {car.get('id')} - {car.get('brand')} {car.get('model')}")
                        return car
                
                # If no exact model match, return the first result
                logger.info(f"No exact model match, using first result: {cars[0].get('id')} - {cars[0].get('brand')} {cars[0].get('model')}")
                return cars[0]
                
            # If not found, try with partial matching
            search_params["partial_match"] = True
            search_results = await self.car_service.search_cars_from_csv(**search_params)
            cars = search_results.get("items", [])
            
            if cars and len(cars) > 0:
                logger.info(f"Found with partial match: {cars[0].get('id')} - {cars[0].get('brand')} {cars[0].get('model')}")
                return cars[0]
                
            # If still not found, try a general search with just the brand
            search_params = {
                "make": brand,
                "page": 1,
                "page_size": 10,
                "partial_match": False
            }

            search_results = await self.car_service.search_cars_from_csv(**search_params)
            cars = search_results.get("items", [])

            # If we have brand matches, return the first one
            if cars and len(cars) > 0:
                logger.info(f"No model match, using brand match: {cars[0].get('id')} - {cars[0].get('brand')} {cars[0].get('model')}")
                return cars[0]

            # If all else fails, return None to indicate no car found
            logger.warning(f"No car found for {brand} {model}")
            return None

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
                "sports": ["sports", "performance", "fast"]
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