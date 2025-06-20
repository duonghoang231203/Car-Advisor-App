#!/usr/bin/env python3
"""
Create enhanced car database tables with comprehensive specifications
"""

import logging
from sqlalchemy import create_engine
from app.core.database import get_database_url
from app.db.enhanced_car_models import Base

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_enhanced_tables():
    """Create all enhanced car tables"""
    try:
        engine = create_engine(get_database_url())
        
        logger.info("Creating enhanced car database tables...")
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("Enhanced car tables created successfully!")
        
        # Log the tables that were created
        logger.info("Created tables:")
        for table_name in Base.metadata.tables.keys():
            logger.info(f"  - {table_name}")
            
    except Exception as e:
        logger.error(f"Error creating enhanced tables: {e}")
        raise

if __name__ == "__main__":
    create_enhanced_tables() 