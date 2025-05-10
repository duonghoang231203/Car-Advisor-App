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
            else:
                # Fallback to OpenAI
                self.llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    openai_api_key=settings.OPENAI_API_KEY,
                    temperature=0.7
                )
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            # Fallback to a mock LLM for safety
            self.llm = None

    def initialize_qa_chain(self):
        """Initialize the QA chain"""
        try:
            template = """
            You are a car advisor helping customers choose cars. Use the following context to answer the question.

            Context: {context}

            Question: {query}

            Provide a helpful, informative response based on the context. If you don't know the answer, say that you don't know.
            """

            prompt = ChatPromptTemplate.from_template(template)

            # Create the QA chain using RunnableSequence
            self.qa_chain = prompt | self.llm
        except Exception as e:
            logger.error(f"Error initializing QA chain: {e}")
            self.qa_chain = None

    async def process_query(self, query, k=10):
        """Process a user query and return a response with car suggestions"""
        try:
            # Step 1: Get relevant car documents
            docs = await vector_db.similarity_search(query, k=k)
            context = "\n".join([doc.page_content for doc in docs]) if docs else ""

            # Step 2: Extract relevant cars for suggestions with diversity
            suggestions = []
            selected_brands = set()
            selected_models = set()
            max_suggestions = 6  # Increased from 3 to 6

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

            # Step 3: If no suggestions were found via vector search, try direct category match
            if not suggestions:
                logger.info("No suggestions from vector search, using fallback mechanism")
                suggestions = await self._get_fallback_suggestions(query)

                # If we got fallback suggestions, update the context with information about these cars
                if suggestions:
                    car_desc = []
                    for car in suggestions:
                        car_desc.append(f"{car.get('brand', '')} {car.get('model', '')} (MSRP: ${car.get('price', 0):,.2f})")

                    additional_context = f"\nRelevant cars for this query include: {', '.join(car_desc)}.\n"
                    context = context + additional_context if context else additional_context

            # Step 4: Generate an explanation based on the suggestions
            explanation = await self._generate_explanation(suggestions, query)

            # Step 5: Now generate the answer using the QA chain with both the document context and suggestion info
            suggestion_context = ""
            if suggestions:
                car_names = [f"{s.get('brand', '')} {s.get('model', '')}" for s in suggestions]
                suggestion_context = f"\nBased on the query, the following cars could be recommended: {', '.join(car_names)}.\n{explanation}\n"

            # Combine original context with suggestion context
            enhanced_context = f"{context}\n{suggestion_context}".strip()

            # Get answer using the QA chain with enhanced context that includes suggestions
            response = await self.qa_chain.ainvoke({"context": enhanced_context, "query": query})

            logger.info(f"Query: {query}")
            logger.info(f"Found {len(suggestions)} suggestions")

            return {
                "response": response if isinstance(response, str) else response.content,
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

    async def _generate_explanation(self, suggestions, query):
        """Generate an explanation for why these cars were suggested"""
        try:
            if not suggestions:
                return "No specific car suggestions were found based on your query."

            # Create a prompt for explanation
            car_descriptions = []
            for car in suggestions:
                brand = car.get('brand', '')
                model = car.get('model', '')
                if brand and model:
                    car_descriptions.append(f"{brand} {model}")

            if not car_descriptions:
                return "No specific car suggestions were found."

            explanation_prompt = f"""
            The user asked: "{query}"

            Based on this query, the following cars were suggested:
            {', '.join(car_descriptions)}

            Provide a brief explanation (1-2 sentences) for why these cars might be suitable for the user's needs.
            """

            # Get response from LLM
            response = await self.llm.ainvoke(explanation_prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return "These cars match your search criteria."

rag_service = RAGService()