from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.database import mysql
from app.api import cars, user as users, auth, chat, query, monitoring
from app.core.logging import logger
from app.core.monitoring import MonitoringMiddleware
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

# Add monitoring middleware
app.add_middleware(MonitoringMiddleware)

# Include routers
app.include_router(cars.router, prefix="/api/cars", tags=["cars"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(query.router, prefix="/api/query", tags=["query"])
# CSV data API has been consolidated into the cars API
app.include_router(monitoring.router, prefix="/api/monitoring", tags=["monitoring"])

@app.on_event("startup")
async def startup():
    logger.info("Starting up application...")

    # Initialize database connection with error handling
    try:
        logger.info("Initializing database connection...")
        await mysql.connect()

        # Create database tables
        from app.core.database import Base, engine
        # Import all models to register them with SQLAlchemy
        try:
            import app.db.models  # This imports all models
        except ImportError as e:
            logger.warning(f"Could not import models: {str(e)}")
            logger.warning("Database tables may not be created properly")

        try:
            async with engine.begin() as conn:
                # Create tables if they don't exist
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {str(e)}")
            logger.warning("Application will continue without database tables")

        logger.info("Database connection established successfully")
    except Exception as e:
        logger.error(f"Failed to initialize SQL service: {str(e)}")
        logger.warning("SQL service will be unavailable, but the application will continue to run.")

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