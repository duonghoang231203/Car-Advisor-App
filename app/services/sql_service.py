from llama_index.core import SQLDatabase
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.core.objects import SQLTableNodeMapping, ObjectIndex
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from app.db.mysql import mysql
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class SQLService:
    def __init__(self):
        self.sql_database = None
        self.query_engine = None
        self.initialize()
    
    def initialize(self):
        try:
            # Initialize SQLDatabase with MySQL connection
            self.sql_database = SQLDatabase.from_uri(
                f"mysql+pymysql://{mysql.MYSQL_USER}:{mysql.MYSQL_PASSWORD}@{mysql.MYSQL_HOST}:{mysql.MYSQL_PORT}/{mysql.MYSQL_DB_NAME}"
            )
            
            # Get all table names
            table_names = self.sql_database.get_table_names()
            
            # Create table node mapping
            table_node_mapping = SQLTableNodeMapping(self.sql_database)
            
            # Create object index
            table_schema_objs = []
            for table_name in table_names:
                table_schema_objs.append(table_node_mapping.to_node(table_name))
            
            obj_index = ObjectIndex.from_objects(
                table_schema_objs,
                table_node_mapping,
                VectorStoreIndex,
            )
            
            # Create query engine
            self.query_engine = SQLTableRetrieverQueryEngine(
                self.sql_database,
                obj_index.as_retriever(similarity_top_k=1),
            )
            
            logger.info("SQL service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SQL service: {str(e)}")
            raise ConnectionError(f"Could not initialize SQL service: {str(e)}")
    
    async def execute_nl_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a natural language query and return the results
        """
        try:
            # Convert natural language to SQL and execute
            response = self.query_engine.query(query)
            
            # Convert response to list of dictionaries
            results = []
            if hasattr(response, 'response') and response.response:
                # If the response is a string representation of the results
                results = [{"result": response.response}]
            elif hasattr(response, 'source_nodes'):
                # If the response contains source nodes with metadata
                for node in response.source_nodes:
                    if hasattr(node, 'metadata'):
                        results.append(node.metadata)
            
            return results
        except Exception as e:
            logger.error(f"Error executing natural language query: {str(e)}")
            raise Exception(f"Error executing query: {str(e)}")
    
    async def execute_sql_query(self, query: str, params: List[Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a raw SQL query and return the results
        """
        try:
            results = await mysql.execute_query(query, params)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error executing SQL query: {str(e)}")
            raise Exception(f"Error executing query: {str(e)}")

sql_service = SQLService() 