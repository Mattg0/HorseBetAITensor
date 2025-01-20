import mysql.connector
import sqlite3
import json
from collections import defaultdict
from decimal import Decimal


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

        return data, columns

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None, None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def convert_to_int(value):
    """Convert Decimal to int or return the value as is."""
    if isinstance(value, Decimal):
        return int(value)  # Convert Decimal to int
    elif isinstance(value, (int, float)):
        return int(value)  # Convert float to int
    return value


def transform_data(data):
    """Transform the fetched data into a structured format."""
    course_results = defaultdict(list)

    # Process each row of data
    for row in data:
        comp = convert_to_int(row[0])  # ID de la course
        cl = convert_to_int(row[1])  # Ordre d'arrivée
        numero = convert_to_int(row[2])  # Numéro du cheval
        idche = convert_to_int(row[3])  # ID du cheval

        # Ajouter le résultat à la liste pour cette course
        course_results[comp].append({
            'narrivee': cl,
            'cheval': numero,
            'idche': idche
        })

    # Sérialiser les résultats pour chaque course
    serialized_results = []
    for comp, results in course_results.items():
        def transform_narrivee(value):
            try:
                return int(value)  # Essayer de convertir en entier
            except (ValueError, TypeError):  # En cas d'erreur, retourner 99
                return 99

        # Appliquer la transformation à chaque résultat
        for result in results:
            result['narrivee'] = transform_narrivee(result['narrivee'])

        # Trier les résultats par ordre d'arrivée (cl)
        results.sort(key=lambda x: (x['narrivee'] != 99, x['narrivee']))

        serialized_results.append({
            'comp': comp,
            'ordre_arrivee': json.dumps(results)  # Sérialiser en JSON
        })

    return serialized_results


def insert_data_into_sqlite(sqlite_db, serialized_results):
    """Insert transformed data into SQLite database."""
    conn = sqlite3.connect(sqlite_db)
    cursor = conn.cursor()

    # Prepare the insert statement
    insert_query = '''
    INSERT INTO Resultats (comp, ordre_arrivee)
    VALUES (?, ?)
    '''

    for result in serialized_results:
        cursor.execute(insert_query, (result['comp'], result['ordre_arrivee']))

    conn.commit()
    conn.close()
    print("Data inserted into SQLite successfully.")


def main():
    # MySQL connection parameters
    mysql_host = "127.0.0.1"  # Change this to your MySQL host
    mysql_user = "turfai"  # Change this to your MySQL username
    mysql_password = "welcome123"  # Change this to your MySQL password
    mysql_db = "pturf2024"  # Change this to your MySQL database name
    mysql_query = "SELECT comp, cl, numero, idche FROM cachedate;"  # Your MySQL query

    # SQLite database file
    sqlite_db = "../data/lite_hippique.db"

    # Fetch data from MySQL
    data, columns = fetch_data_from_mysql(mysql_host, mysql_user, mysql_password, mysql_db, mysql_query)

    if data is not None and columns is not None:
        # Transform the data into the desired structure
        serialized_results = transform_data(data)

        # Insert data into SQLite
        insert_data_into_sqlite(sqlite_db, serialized_results)


if __name__ == "__main__":
    main()