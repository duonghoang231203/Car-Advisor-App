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
            self.embeddings = OpenAIEmbeddings(openai_api_key=settings.LLM_API_KEY)
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
        """Perform similarity search on the vector database"""
        return self.db.similarity_search(query, k=k)
    
    async def add_texts(self, texts, metadatas=None):
        """Add texts to the vector database"""
        return self.db.add_texts(texts=texts, metadatas=metadatas)

vector_db = VectorDB() 