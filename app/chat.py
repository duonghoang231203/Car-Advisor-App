# chatbot.py - Updated version with error handling and alternatives

from sentence_transformers import SentenceTransformer
import chromadb
from cachetools import cached, TTLCache
from transformers import pipeline
import openai
from sqlalchemy import create_engine, Column, Integer, String, Sequence
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import logging
from typing import Optional

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# Initialize Sentence Transformer for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB for vector storage
client = chromadb.Client()
collection = client.create_collection("car_data")

# Initialize cache
cache = TTLCache(maxsize=100, ttl=300)

# Initialize OpenAI API
openai.api_key = os.getenv("LLM_API_KEY")

# Initialize local NLP pipeline as fallback
nlp_pipeline = pipeline('text-generation', model='gpt2')

# Initialize SQLAlchemy for conversation memory
Base = declarative_base()

class Conversation(Base):
    __tablename__ = 'conversations'
    id = Column(Integer, Sequence('conversation_id_seq'), primary_key=True)
    user_id = Column(Integer)
    message = Column(String)
    response = Column(String)

engine = create_engine('sqlite:///:memory:')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Function to generate embeddings
def generate_embeddings(data):
    return model.encode(data)

# Function to add embeddings to ChromaDB
def add_to_collection(data, metadata):
    embeddings = generate_embeddings(data)
    collection.add(embeddings, metadata=metadata)

# Function to generate SQL query
def generate_sql(query):
    return f"SELECT * FROM cars WHERE mileage > {query['mileage']}"

# Fallback response generator
def get_local_response(prompt):
    try:
        result = nlp_pipeline(prompt, max_length=100)
        return result[0]['generated_text']
    except Exception as e:
        logger.error(f"Local model error: {e}")
        return "I'm unable to process your request at the moment."

# Function to get AI response with fallback
def get_ai_response(prompt):
    try:
        # Try using OpenAI with better system prompt
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a car sales assistant that provides helpful, accurate, and concise information about vehicles. Only discuss automobiles and relevant purchase considerations. Keep responses focused on car-related queries and under 150 words."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0.7
        )
        return response.choices[0].message['content']
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        # Fallback to local model
        return get_local_response(prompt)

# Request model
class QueryRequest(BaseModel):
    query: str
    user_id: Optional[int] = None

# Function to handle user queries
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        response_text = get_ai_response(request.query)
        
        # Store conversation if user_id is provided
        if request.user_id:
            new_conversation = Conversation(
                user_id=request.user_id,
                message=request.query,
                response=response_text
            )
            session.add(new_conversation)
            session.commit()
            
        return {"result": response_text}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Example usage
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chat:app", host="0.0.0.0", port=8000, reload=True, log_level="info")