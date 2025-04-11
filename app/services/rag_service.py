from typing import Dict, List, Any
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableSequence
from app.db.vector_store import vector_db
from app.config import settings
from app.models.car import Car
import json

class RAGService:
    def __init__(self):
        """Initialize the RAG service"""
        self.initialize_llm()
        self.qa_chain = None
        self.initialize_qa_chain()
    
    def initialize_llm(self):
        """Initialize the language model"""
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
    
    def initialize_qa_chain(self):
        """Initialize the QA chain"""
        template = """
        You are a car advisor helping customers choose cars. Use the following context to answer the question.
        
        Context: {context}
        
        Question: {query}
        
        Provide a helpful, informative response based on the context. If you don't know the answer, say that you don't know.
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create the QA chain using RunnableSequence
        self.qa_chain = prompt | self.llm
    
    async def process_query(self, query, k=5):
        """Process a user query and return a response with car suggestions"""
        # Get relevant car documents
        docs = await vector_db.similarity_search(query, k=k)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Get answer using the QA chain
        response = await self.qa_chain.ainvoke({"context": context, "query": query})
        
        # Extract relevant cars for suggestions
        suggestions = []
        if docs:
            for doc in docs[:3]:  # Only use top 3 for suggestions
                if doc.metadata and "car_id" in doc.metadata:
                    car_id = doc.metadata["car_id"]
                    car_data = await self._get_car_data(car_id)
                    if car_data:
                        suggestions.append(car_data)
        
        # Get explanations for suggestions
        explanation = await self._generate_explanation(suggestions, query)
        
        return {
            "response": response if isinstance(response, str) else response.content,
            "suggestions": suggestions,
            "explanation": explanation
        }
    
    async def _get_car_data(self, car_id):
        """Get car data by ID"""
        from app.db.mongodb import mongodb
        
        car_data = await mongodb.db.cars.find_one({"_id": car_id})
        if car_data:
            return Car(**car_data).dict()
        return None

    async def _generate_explanation(self, suggestions, query):
        """Generate an explanation for why these cars were suggested"""
        if not suggestions:
            return "No specific car suggestions were found based on your query."
        
        # Create a prompt for explanation
        explanation_prompt = f"""
        The user asked: "{query}"
        
        Based on this query, the following cars were suggested:
        {', '.join([f"{car['brand']} {car['model']} ({car['year']})" for car in suggestions])}
        
        Provide a brief explanation (1-2 sentences) for why these cars might be suitable for the user's needs.
        """
        
        # Get response from LLM
        response = await self.llm.ainvoke(explanation_prompt)
        return response.content

rag_service = RAGService() 