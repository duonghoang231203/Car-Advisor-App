import asyncio
import argparse
from app.db.migrations import create_tables, drop_tables
from app.db.mysql import connect_to_mysql, close_mysql_connection

async def main():
    parser = argparse.ArgumentParser(description='Database migration tool')
    parser.add_argument('--action', choices=['create', 'drop'], required=True,
                      help='Action to perform: create or drop tables')
    
    args = parser.parse_args()
    
    try:
        # Connect to MySQL
        await connect_to_mysql()
        
        # Perform the requested action
        if args.action == 'create':
            await create_tables()
            print("Tables created successfully")
        elif args.action == 'drop':
            await drop_tables()
            print("Tables dropped successfully")
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        # Close the connection
        await close_mysql_connection()

if __name__ == "__main__":
    asyncio.run(main()) 