import os
from typing import List, Union
from pydantic import field_validator, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # MySQL settings
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DB_NAME: str = "car_advisor"
    DATABASE_URL: str = ""
    
    # App settings
    APP_NAME: str = "Car Advisor API"
    API_PREFIX: str = "/api"
    DEBUG: bool = False
    
    # Security settings
    SECRET_KEY: str = "your-secret-key"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # CORS settings
    CORS_ORIGINS: Union[str, List[str]] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="List of origins that are allowed to make cross-site requests"
    )
    
    # LLM settings
    LLM_TYPE: str = "openai"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gpt-3.5-turbo"
    
    # Vector DB settings
    VECTOR_DB_TYPE: str = "chroma"  # or "pinecone"
    VECTOR_DB_API_KEY: str = ""  # Only needed for Pinecone
    
    # Speech-to-Text settings
    STT_PROVIDER: str = "google"
    STT_API_KEY: str = ""
    
    @field_validator('DEBUG', mode='before')
    @classmethod
    def parse_debug(cls, v):
        if isinstance(v, str):
            return v.lower() == "true"
        return v
    
    @field_validator('ACCESS_TOKEN_EXPIRE_MINUTES', mode='before')
    @classmethod
    def parse_expire_minutes(cls, v):
        if isinstance(v, str):
            return int(v)
        return v
    
    @field_validator('CORS_ORIGINS', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            if v == "*":
                return ["*"]
            if not isinstance(v, str):
                return v
            if v.startswith("["):
                try:
                    import json
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        populate_by_name = True
        extra = "ignore"
    
    def model_post_init(self, __context=None) -> None:
        # Apply environment variable overrides
        self.DEBUG = os.getenv("DEBUG", "False").lower() == "true"
        self.MYSQL_HOST = os.getenv("MYSQL_HOST", self.MYSQL_HOST)
        self.MYSQL_PORT = int(os.getenv("MYSQL_PORT", self.MYSQL_PORT))
        self.MYSQL_USER = os.getenv("MYSQL_USER", self.MYSQL_USER)
        self.MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", self.MYSQL_PASSWORD)
        self.MYSQL_DB_NAME = os.getenv("MYSQL_DB_NAME", self.MYSQL_DB_NAME)
        self.SECRET_KEY = os.getenv("SECRET_KEY", self.SECRET_KEY)
        
        # Create DATABASE_URL from MySQL settings
        self.DATABASE_URL = f"mysql+aiomysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DB_NAME}"
        
        access_token_expire = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")
        if access_token_expire:
            self.ACCESS_TOKEN_EXPIRE_MINUTES = int(access_token_expire)
        
        self.VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", self.VECTOR_DB_TYPE)
        self.VECTOR_DB_API_KEY = os.getenv("VECTOR_DB_API_KEY", self.VECTOR_DB_API_KEY)
        
        self.STT_PROVIDER = os.getenv("STT_PROVIDER", self.STT_PROVIDER)
        self.STT_API_KEY = os.getenv("STT_API_KEY", self.STT_API_KEY)

settings = Settings() 