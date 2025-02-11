import mysql.connector
import json
from decimal import Decimal
import sys
sys.path.append('../../')

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
    """Transform the fetched data into a structured format for a single course."""
    if not data:
        return None  # Return None if no data is available

    # Create a dictionary to hold the course data
    course_info = {}
    participants = []

    # Identify the indices of the relevant columns
    comp_index = columns.index('id')
    hippo_index = columns.index('hippo')
    jour_index = columns.index('jour')
    meteo_index = columns.index('meteo')
    dist_index = columns.index('dist')
    corde_index = columns.index('corde')
    natpis_index = columns.index('natpis')
    pistegp_index = columns.index('pistegp')
    typec_index = columns.index('typec')
    temperature_index = columns.index('temperature')
    forceVent_index = columns.index('forceVent')
    directionVent_index = columns.index('directionVent')
    nebulosite_index = columns.index('nebulositeLibelleCourt')

    # Process each row of data (assuming all rows belong to the same course)
    for row in data:
        # Store course info (only keep the first occurrence)
        if not course_info:
            course_info = {
                'comp': row[comp_index],
                'hippo': row[hippo_index],
                'jour': row[jour_index],
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

        # Add participant info
        participant_info = {
            'idche': convert_decimal(row[columns.index('idche')]),
            'cheval': row[columns.index('cheval')],
            'numero': int(row[columns.index('numero')]),
            'age': int(row[columns.index('age')]),
            'musiqueche': row[columns.index('musiqueche')],
            'idJockey': row[columns.index('idJockey')],
            'musiquejoc': row[columns.index('musiquejoc')],
            'idEntraineur': convert_decimal(row[columns.index('idEntraineur')]),
            'cotedirect': float(row[columns.index('cotedirect')]),
        }
        participants.append(participant_info)

    return {
        'course_info': course_info,
        'participants': participants
    }

def main(comp):
    # MySQL connection parameters
    mysql_host = "localhost"  # Change this to your MySQL host
    mysql_user = "turfai"  # Change this to your MySQL username
    mysql_password = "welcome123"  # Change this to your MySQL password
    mysql_db = "pturf2015"  # Change this to your MySQL database name
    mysql_query = f"""
    SELECT caractrap.id, caractrap.jour, caractrap.hippo, caractrap.meteo, caractrap.dist,caractrap.typec,
           caractrap.corde, caractrap.natpis, caractrap.pistegp, caractrap.arriv,
           caractrap.temperature, caractrap.forceVent, caractrap.directionVent,
           caractrap.nebulositeLibelleCourt, cachedate.idche, cachedate.cheval,
           cachedate.numero,cachedate.age, musiqueche, cachedate.idJockey, musiquejoc, cachedate.idEntraineur,cachedate.cotedirect
    FROM caractrap
    INNER JOIN cachedate ON caractrap.id = cachedate.comp WHERE cachedate.comp = {comp}
    """  # Use the comp parameter in the query

    # Fetch data from MySQL
    data, columns = fetch_data_from_mysql(mysql_host, mysql_user, mysql_password, mysql_db, mysql_query)

    if data is not None and columns is not None:
        # Transform the data into the desired structure
        course_data = transform_data(data, columns)

        # Convert course_data to JSON format
        return json.dumps(course_data)  # Retourner les données au format JSON

    return None  # Retourner None si aucune donnée n'est récupérée