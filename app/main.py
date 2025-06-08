from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.database import mysql
from app.api import cars, user as users, auth, chat, monitoring
from app.core.logging import logger
from app.core.monitoring import MonitoringMiddleware
import logging
import uvicorn
from app.config import settings
import asyncio
from contextlib import asynccontextmanager
import signal
from fastapi.openapi.utils import get_openapi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Global flag for graceful shutdown
shutdown_event = asyncio.Event()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up application...")
    try:
        # Initialize any resources here
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
        yield
    finally:
        # Shutdown
        logger.info("Shutting down application...")
        try:
            # Set shutdown event
            shutdown_event.set()
            
            # Wait for a short time to allow ongoing requests to complete
            await asyncio.sleep(1)
            
            # Cleanup database connection
            await mysql.disconnect()
            logger.info("Database connection closed successfully")
            
            # Cancel remaining tasks
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            if tasks:
                for task in tasks:
                    task.cancel()
                try:
                    await asyncio.wait(tasks, timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning("Some tasks did not complete in time")
                except Exception as e:
                    logger.error(f"Error during task cleanup: {str(e)}")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")

def handle_sigterm(signum, frame):
    """Handle SIGTERM signal"""
    logger.info("Received SIGTERM signal")
    if not shutdown_event.is_set():
        shutdown_event.set()

# Register signal handlers
signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

app = FastAPI(
    title=settings.APP_NAME,
    description="API for car rental management system",
    version="1.0.0",
    docs_url="/docs",  # Enable Swagger UI at /docs
    lifespan=lifespan,
    swagger_ui_init_oauth={
        "usePkceWithAuthorizationCodeGrant": True,
    },
    openapi_tags=[
        {"name": "Authentication", "description": "Authentication operations"},
        {"name": "Users", "description": "User management operations"},
        {"name": "Cars", "description": "Car operations"},
        {"name": "Chat", "description": "Chat operations"},
        {"name": "Monitoring", "description": "Monitoring operations"},
    ]
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add monitoring middleware
app.add_middleware(MonitoringMiddleware)

# Include routers with explicit paths
app.include_router(cars.router, prefix=f"{settings.API_PREFIX}/cars", tags=["Cars"])
app.include_router(users.router, prefix=f"{settings.API_PREFIX}/users", tags=["Users"])
app.include_router(auth.router, prefix=f"{settings.API_PREFIX}/auth", tags=["Authentication"])
app.include_router(chat.router, prefix=f"{settings.API_PREFIX}/chat", tags=["Chat"])
app.include_router(monitoring.router, prefix=f"{settings.API_PREFIX}/monitoring", tags=["Monitoring"])

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to Car Rental API", "docs": "/docs"}

@app.get("/health")
async def health():
    logger.info("Health check endpoint accessed")
    return {"status": "ok"}

# Add security scheme for Bearer token
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=settings.APP_NAME,
        version="1.0.0",
        description="API for car rental management system",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    logger.info("Starting development server...")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        timeout_keep_alive=30,
        timeout_graceful_shutdown=10
    )