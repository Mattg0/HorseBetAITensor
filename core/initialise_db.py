import sqlite3
from core.db_helper import create_connection, initialize_db

def main():
    database = "data/hippique.db"
    conn = create_connection(database)
    initialize_db(conn)
    conn.close()
    print("Database initialized successfully.")

if __name__ == "__main__":
    main()