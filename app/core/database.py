from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings
import logging
from typing import Generator
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=True,
    pool_pre_ping=True
)

# Create async session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Create base class for models
Base = declarative_base()

class MySQL:
    def __init__(self):
        self.engine = engine
        self.session = SessionLocal

    async def connect(self):
        try:
            # Test the connection
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            logger.info("Connected to MySQL database")
        except Exception as e:
            logger.error(f"Failed to connect to MySQL: {str(e)}")
            raise

    async def disconnect(self):
        await self.engine.dispose()
        logger.info("Disconnected from MySQL database")

    async def get_session(self) -> Generator[AsyncSession, None, None]:
        session = self.session()
        try:
            yield session
        finally:
            await session.close()

mysql = MySQL()

def get_database_url():
    """Get database URL for synchronous connections"""
    # Convert async database URL to sync URL for SQLAlchemy ORM
    return settings.DATABASE_URL.replace("mysql+aiomysql://", "mysql+pymysql://")