import aiomysql
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class MySQL:
    def __init__(self):
        self.pool = None
        self.initialize()
    
    def initialize(self):
        try:
            # Connection pool will be created on first use
            logger.info("MySQL connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MySQL: {str(e)}")
            raise ConnectionError(f"Could not initialize MySQL: {str(e)}")
    
    async def get_pool(self):
        if not self.pool:
            self.pool = await aiomysql.create_pool(
                host=settings.MYSQL_HOST,
                port=settings.MYSQL_PORT,
                user=settings.MYSQL_USER,
                password=settings.MYSQL_PASSWORD,
                db=settings.MYSQL_DB_NAME,
                autocommit=True,
                charset='utf8mb4',
                cursorclass=aiomysql.DictCursor
            )
        return self.pool
    
    async def execute_query(self, query, params=None):
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                return await cur.fetchall()
    
    async def execute_many(self, query, params_list):
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(query, params_list)
                return cur.rowcount
    
    async def close(self):
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            logger.info("Closed MySQL connection pool")

mysql = MySQL()

async def connect_to_mysql():
    await mysql.get_pool()
    logger.info(f"Connected to MySQL: {settings.MYSQL_HOST}")

async def close_mysql_connection():
    await mysql.close()
    logger.info("Closed MySQL connection") 