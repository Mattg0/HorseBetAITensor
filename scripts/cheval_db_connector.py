import mysql.connector
import sqlite3
from core.db_helper import create_connection

import mysql.connector
import sqlite3


def fetch_data_from_mysql(mysql_host, mysql_user, mysql_password, mysql_db, mysql_query):
    """Connect to MySQL and fetch data based on the provided query."""
    try:
        # Connect to MySQL
        connection = mysql.connector.connect(
            host=mysql_host,
            user=mysql_user,
            password=mysql_password,
            database=mysql_db
        )

        cursor = connection.cursor()
        cursor.execute(mysql_query)
        data = cursor.fetchall()  # Fetch all rows from the executed query
        columns = [column[0] for column in cursor.description]  # Get column names

        # Filter out the 'jour' column
        filtered_columns = [col for col in columns if col != 'jour']
        return data, filtered_columns

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None, None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def remove_duplicates(data, id_index):
    """Remove duplicates based on the idche key, keeping only the first occurrence."""
    seen = set()
    unique_data = []

    for row in data:
        idche_value = row[id_index]
        if idche_value not in seen:
            seen.add(idche_value)
            unique_data.append(row)

    return unique_data

def insert_data_into_sqlite(sqlite_db, data, columns):
    """Insert fetched data into SQLite database."""
    conn = sqlite3.connect(sqlite_db)
    cursor = conn.cursor()

    # Prepare the insert statement, excluding the 'jour' column
    placeholders = ', '.join(['?'] * len(columns))  # Create placeholders for SQLite
    insert_query = f"INSERT OR IGNORE INTO Cheval ({', '.join(columns)}) VALUES ({placeholders})"

    # Insert each row of data, excluding the 'jour' column
    filtered_data = []
    for row in data:
        filtered_row = [row[columns.index(col)] for col in columns if col != 'jour']
        filtered_data.append(tuple(filtered_row))

    cursor.executemany(insert_query, filtered_data)

    conn.commit()
    conn.close()
    print("Data inserted into SQLite successfully.")



def main():
    # MySQL connection parameters
    mysql_host = "127.0.0.1"  # Change this to your MySQL host
    mysql_user = "turfai"  # Change this to your MySQL username
    mysql_password = "welcome123"  # Change this to your MySQL password
    mysql_db = "pturf2014"  # Change this to your MySQL database name
    mysql_query = "SELECT idche ,cheval as nom,sexe,age,jour FROM cachedate ORDER BY jour DESC;"  # Change this to your MySQL query

    # SQLite database file
    sqlite_db = "../data/hippique.db"

    # Fetch data from MySQL
    data, columns = fetch_data_from_mysql(mysql_host, mysql_user, mysql_password, mysql_db, mysql_query)

    if data is not None and columns is not None:
        # Remove duplicates based on the idche column (assuming it's the first column)
        idche_index = columns.index('idche')  # Get the index of the idche column
        unique_data = remove_duplicates(data, idche_index)
        # Insert data into SQLite
        insert_data_into_sqlite(sqlite_db, unique_data, columns)


if __name__ == "__main__":
    main()