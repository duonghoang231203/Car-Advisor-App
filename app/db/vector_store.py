import os
from langchain_chroma import Chroma
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import settings

class VectorDB:
    def __init__(self):
        self.db = None
        self.embeddings = None
        self.initialize()
    
    def initialize(self):
        # Initialize embeddings
        if settings.LLM_TYPE == "openai":
            self.embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
        else:
            # Default to a good open-source embedding model
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize vector store
        if settings.VECTOR_DB_TYPE == "chroma":
            persist_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "embeddings")
            self.db = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)
        elif settings.VECTOR_DB_TYPE == "pinecone":
            import pinecone
            pinecone.init(api_key=settings.VECTOR_DB_API_KEY)
            self.db = Pinecone.from_existing_index(index_name="car-advisor", embedding=self.embeddings)
    
    async def similarity_search(self, query, k=5):
        """
        Perform similarity search on the vector database with diversification.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List of documents that are semantically similar to the query
        """
        try:
            import random
            from langchain.schema import Document
            
            # Basic error checking
            if not query or not self.db:
                return []
                
            # First, get more results than needed to allow for diversification
            fetch_k = min(k * 3, 30)  # Fetch 3x results but cap at 30 to avoid excessive computation
            
            # Perform the similarity search
            results = self.db.similarity_search(query, k=fetch_k)
            
            if not results:
                return []
                
            # Log the number of results found
            print(f"Vector search found {len(results)} results for query: {query}")
            
            # Extract document IDs to ensure diversity
            doc_ids = set()
            car_ids = set()
            car_brands = set()
            diverse_results = []
            
            # First pass: Add highly relevant results while maintaining diversity
            for doc in results:
                # Skip documents without metadata
                if not hasattr(doc, 'metadata') or not doc.metadata:
                    continue
                    
                # Get document identifier
                doc_id = doc.metadata.get("doc_id", None)
                car_id = doc.metadata.get("id", doc.metadata.get("car_id", None))
                car_brand = None
                
                # Try to extract brand information for diversity
                if "brand" in doc.metadata:
                    car_brand = doc.metadata["brand"]
                elif "content" in doc.metadata and isinstance(doc.metadata["content"], str):
                    # Try to extract brand from content
                    content = doc.metadata["content"]
                    common_brands = ["Toyota", "Honda", "Ford", "Chevrolet", "BMW", "Mercedes", 
                                    "Audi", "Lexus", "Nissan", "Hyundai", "Kia", "Subaru"]
                    for brand in common_brands:
                        if brand in content:
                            car_brand = brand
                            break
                            
                # Skip if we've already included this document or car
                if doc_id and doc_id in doc_ids:
                    continue
                    
                # Limit results from the same car ID or brand for diversity
                if car_id and car_id in car_ids and len(car_ids) >= 3:
                    continue
                    
                if car_brand and car_brand in car_brands and len(car_brands) >= 3:
                    continue
                    
                # Add document and track IDs
                diverse_results.append(doc)
                if doc_id:
                    doc_ids.add(doc_id)
                if car_id:
                    car_ids.add(car_id)
                if car_brand:
                    car_brands.add(car_brand)
                    
                # Stop once we have enough diverse results
                if len(diverse_results) >= k:
                    break
                    
            # If we don't have enough results, add some of the remaining ones
            if len(diverse_results) < k and len(results) > len(diverse_results):
                remaining = [doc for doc in results if doc not in diverse_results]
                # Randomly select from remaining to add diversity
                random.shuffle(remaining)
                diverse_results.extend(remaining[:k - len(diverse_results)])
                
            # Ensure we return at most k results
            return diverse_results[:k]
            
        except Exception as e:
            import logging
            logging.error(f"Error in vector search: {e}")
            # Fallback to basic search in case of error
            try:
                return self.db.similarity_search(query, k=k)
            except:
                return []
    
    async def add_texts(self, texts, metadatas=None):
        """Add texts to the vector database"""
        return self.db.add_texts(texts=texts, metadatas=metadatas)

vector_db = VectorDB() 