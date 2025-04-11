from app.db.mysql import mysql
import logging

logger = logging.getLogger(__name__)

async def create_tables():
    try:
        # Create cars table
        await mysql.execute_query("""
            CREATE TABLE IF NOT EXISTS cars (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                brand VARCHAR(100) NOT NULL,
                model VARCHAR(100) NOT NULL,
                year INT NOT NULL,
                price DECIMAL(10, 2) NOT NULL,
                car_condition VARCHAR(50) NOT NULL,
                car_type VARCHAR(50) NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_brand (brand),
                INDEX idx_price (price),
                INDEX idx_year (year)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)

        # Create car_specifications table
        await mysql.execute_query("""
            CREATE TABLE IF NOT EXISTS car_specifications (
                id INT AUTO_INCREMENT PRIMARY KEY,
                car_id INT NOT NULL,
                engine VARCHAR(100),
                transmission VARCHAR(100),
                fuel_type VARCHAR(50),
                mileage DECIMAL(10, 2),
                seating_capacity INT,
                body_type VARCHAR(50),
                FOREIGN KEY (car_id) REFERENCES cars(id) ON DELETE CASCADE,
                INDEX idx_car_id (car_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)

        # Create car_features table
        await mysql.execute_query("""
            CREATE TABLE IF NOT EXISTS car_features (
                id INT AUTO_INCREMENT PRIMARY KEY,
                car_id INT NOT NULL,
                feature_name VARCHAR(100) NOT NULL,
                FOREIGN KEY (car_id) REFERENCES cars(id) ON DELETE CASCADE,
                INDEX idx_car_id (car_id),
                UNIQUE KEY unique_car_feature (car_id, feature_name)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)

        # Create car_images table
        await mysql.execute_query("""
            CREATE TABLE IF NOT EXISTS car_images (
                id INT AUTO_INCREMENT PRIMARY KEY,
                car_id INT NOT NULL,
                image_url VARCHAR(255) NOT NULL,
                FOREIGN KEY (car_id) REFERENCES cars(id) ON DELETE CASCADE,
                INDEX idx_car_id (car_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)

        # Create users table
        await mysql.execute_query("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) NOT NULL UNIQUE,
                email VARCHAR(100) NOT NULL UNIQUE,
                full_name VARCHAR(100),
                phone_number VARCHAR(20),
                hashed_password VARCHAR(255) NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_username (username),
                INDEX idx_email (email)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)

        # Create user_preferences table
        await mysql.execute_query("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                budget_min DECIMAL(10, 2),
                budget_max DECIMAL(10, 2),
                purpose VARCHAR(50),
                passengers INT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                INDEX idx_user_id (user_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)

        # Create user_preferred_brands table
        await mysql.execute_query("""
            CREATE TABLE IF NOT EXISTS user_preferred_brands (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                brand_name VARCHAR(100) NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                INDEX idx_user_id (user_id),
                UNIQUE KEY unique_user_brand (user_id, brand_name)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)

        # Create user_preferred_types table
        await mysql.execute_query("""
            CREATE TABLE IF NOT EXISTS user_preferred_types (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                type_name VARCHAR(100) NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                INDEX idx_user_id (user_id),
                UNIQUE KEY unique_user_type (user_id, type_name)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)

        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        raise

async def drop_tables():
    try:
        # Drop tables in reverse order of creation to handle foreign key constraints
        tables = [
            'user_preferred_types',
            'user_preferred_brands',
            'user_preferences',
            'users',
            'car_images',
            'car_features',
            'car_specifications',
            'cars'
        ]
        
        for table in tables:
            await mysql.execute_query(f"DROP TABLE IF EXISTS {table}")
        
        logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error(f"Error dropping database tables: {str(e)}")
        raise 