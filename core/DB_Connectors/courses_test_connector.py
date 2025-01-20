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


def convert_decimal(value):
    """Convert Decimal to float or return the value as is."""
    if isinstance(value, Decimal):
        return float(value)
    return value


def transform_data(data, columns):
    """Transform the fetched data into a structured format."""
    # Create a dictionary to hold the course data
    course_data = defaultdict(lambda: {
        'participants': [],
        'course_info': {}
    })

    # Identify the indices of the relevant columns
    id_index = columns.index('id')
    arriv_index=columns.index('arriv')
    hippo_index = columns.index('hippo')
    meteo_index = columns.index('meteo')
    dist_index = columns.index('dist')
    corde_index = columns.index('corde')
    natpis_index = columns.index('natpis')
    pistegp_index = columns.index('pistegp')
    typec_index= columns.index('typec')
    temperature_index = columns.index('temperature')
    forceVent_index = columns.index('forceVent')
    directionVent_index = columns.index('directionVent')
    nebulosite_index = columns.index('nebulositeLibelleCourt')

    # Process each row of data
    for row in data:
        comp_id = int(row[id_index])  # Convert to int for comp
        participant_info = {
            'idche': convert_decimal(row[columns.index('idche')]),
            'cheval': row[columns.index('cheval')],
            'age': row[columns.index('age')],
            'numero': int(row[columns.index('numero')]),
            'musiqueche': row[columns.index('musiqueche')],
            'idJockey': convert_decimal(row[columns.index('idJockey')]),
            'musiquejoc': row[columns.index('musiquejoc')],
            'idEntraineur': convert_decimal(row[columns.index('idEntraineur')])
        }

        # Store participant info
        course_data[comp_id]['participants'].append(participant_info)

        # Store course info (only keep the first occurrence)
        if not course_data[comp_id]['course_info']:
            course_data[comp_id]['course_info'] = {
                'arriv': row[arriv_index],
                'hippo': row[hippo_index],
                'meteo': row[meteo_index],
                'dist': convert_decimal(row[dist_index]),
                'corde': row[corde_index],
                'natpis': row[natpis_index],
                'pistegp': row[pistegp_index],
                'typec': row[typec_index],
                'temperature': convert_decimal(row[temperature_index]),
                'forceVent': convert_decimal(row[forceVent_index]),
                'directionVent': row[directionVent_index],
                'nebulosite': row[nebulosite_index]
            }

    return course_data


def insert_data_into_sqlite(sqlite_db, course_data):
    """Insert transformed data into SQLite database."""
    conn = sqlite3.connect(sqlite_db)
    cursor = conn.cursor()

    # Prepare the insert statement
    insert_query = '''
    INSERT INTO Courses_Test (comp,arriv, hippodrome, meteo, dist, corde, natpis, pistegp, typec, temperature, forceVent, directionVent, nebulosite, participants)
    VALUES (?, ?,?, ?, ?,?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''

    for comp_id, info in course_data.items():
        course_info = info['course_info']
        participants_json = json.dumps(info['participants'], default=convert_decimal)  # Serialize participants to JSON

        # Remplacer les valeurs vides par 'N/A'
        arriv = course_info['arriv'] if course_info['arriv'] else 'N/A'
        hippo = course_info['hippo'] if course_info['hippo'] else 'N/A'
        meteo = course_info['meteo'] if course_info['meteo'] else 'N/A'
        dist = course_info['dist'] if course_info['dist'] else 'N/A'
        corde = course_info['corde'] if course_info['corde'] else 'N/A'
        natpis = course_info['natpis'] if course_info['natpis'] else 'N/A'
        pistegp = course_info['pistegp'] if course_info['pistegp'] else 'N/A'
        typec = course_info['typec'] if course_info['typec'] else 'N/A'
        temperature = course_info['temperature'] if course_info['temperature'] else 'N/A'
        forceVent = course_info['forceVent'] if course_info['forceVent'] else 'N/A'
        directionVent = course_info['directionVent'] if course_info['directionVent'] else 'N/A'
        nebulosite = course_info['nebulosite'] if course_info['nebulosite'] else 'N/A'

        cursor.execute(insert_query, (
            comp_id,
            arriv,
            hippo,
            meteo,
            dist,
            corde,
            natpis,
            pistegp,
            typec,
            temperature,
            forceVent,
            directionVent,
            nebulosite,
            participants_json))

    conn.commit()
    conn.close()
    print("Data inserted into SQLite successfully.")


def main():
    # MySQL connection parameters
    mysql_host = "127.0.0.1"  # Change this to your MySQL host
    mysql_user = "turfai"  # Change this to your MySQL username
    mysql_password = "welcome123"  # Change this to your MySQL password
    mysql_db = "pturf2024"  # Change this to your MySQL database name
    mysql_query = """
    SELECT caractrap.id,caractrap.arriv, caractrap.jour, caractrap.hippo, caractrap.meteo, caractrap.dist,
           caractrap.corde, caractrap.natpis, caractrap.pistegp, caractrap.arriv,caractrap.typec,
           caractrap.temperature, caractrap.forceVent, caractrap.directionVent,
           caractrap.nebulositeLibelleCourt, cachedate.idche, cachedate.cheval,
           cachedate.numero, musiqueche, cachedate.idJockey, musiquejoc, cachedate.idEntraineur,cachedate.age
    FROM caractrap
    INNER JOIN cachedate ON caractrap.id = cachedate.comp WHERE caractrap.jour='2024-12-15'
    """  # Change this to your MySQL query

    # SQLite database file
    sqlite_db = "../data/hippique.db"

    # Fetch data from MySQL
    data, columns = fetch_data_from_mysql(mysql_host, mysql_user, mysql_password, mysql_db, mysql_query)

    if data is not None and columns is not None:
        # Transform the data into the desired structure
        course_data = transform_data(data, columns)

        # Insert data into SQLite
        insert_data_into_sqlite(sqlite_db, course_data)


if __name__ == "__main__":
    main()
