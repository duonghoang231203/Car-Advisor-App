import asyncio
import aiomysql
from app.config import settings

async def create_database():
    try:
        # Connect to MySQL server without specifying a database
        conn = await aiomysql.connect(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            charset='utf8mb4'
        )
        
        async with conn.cursor() as cur:
            # Create database if it doesn't exist
            await cur.execute(f"CREATE DATABASE IF NOT EXISTS {settings.MYSQL_DB_NAME}")
            print(f"Database '{settings.MYSQL_DB_NAME}' created successfully")
        
        await conn.close()
        
    except Exception as e:
        print(f"Error creating database: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(create_database()) 