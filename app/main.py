from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.database import mysql
from app.api import cars, user as users, auth, chat, query
from app.core.logging import logger
import logging
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = FastAPI(
    title="Car Rental API",
    description="API for car rental management system",
    version="1.0.0",
    docs_url="/docs",  # Enable Swagger UI at /docs
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(cars.router, prefix="/api/cars", tags=["cars"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(query.router, prefix="/api/query", tags=["query"])

@app.on_event("startup")
async def startup():
    logger.info("Starting up application...")
    logger.info("Initializing database connection...")
    await mysql.connect()

    # Create database tables
    from app.core.database import Base, engine
    # Import all models to register them with SQLAlchemy
    import app.db.models  # This imports all models

    async with engine.begin() as conn:
        # Create tables if they don't exist
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database tables created successfully")
    logger.info("Database connection established successfully")

@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down application...")
    logger.info("Closing database connection...")
    await mysql.disconnect()
    logger.info("Database connection closed successfully")

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to Car Rental API", "docs": "/docs"}

@app.get("/health")
async def health():
    logger.info("Health check endpoint accessed")
    return {"status": "ok"}

if __name__ == "__main__":
    logger.info("Starting development server...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")