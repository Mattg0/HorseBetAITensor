import sqlite3
import json
from decimal import Decimal
from typing import Optional, Dict, Any
import sys
from pathlib import Path

sys.path.append('../../')

from env_setup import setup_environment, get_model_paths,get_database_path


def connect_to_db(db_path: str) -> sqlite3.Connection:
    """Connect to SQLite database with optimized settings."""
    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA journal_mode = WAL')
    conn.execute('PRAGMA cache_size = -2000')
    conn.execute('PRAGMA synchronous = NORMAL')
    return conn


def convert_decimal(value: Any) -> float:
    """Convert Decimal to float or return the value as is."""
    if isinstance(value, Decimal):
        return float(value)
    return float(value) if value is not None else 0.0


def fetch_race_data(db_path: str, comp_id: int) -> Optional[tuple]:
    """Fetch race data from SQLite database."""
    try:
        connection = connect_to_db(db_path)
        cursor = connection.cursor()

        # Query for course info and participants from daily_race table
        query = """
        SELECT 
            dr.comp,
            dr.hippodrome,
            dr.jour,
            dr.meteo,
            dr.dist,
            dr.corde,
            dr.natpis,
            dr.pistegp,
            dr.typec,
            dr.temperature,
            dr.forceVent,
            dr.directionVent,
            dr.nebulosite,
            dr.participants
        FROM daily_races dr
        WHERE dr.comp = ?
        """

        cursor.execute(query, (comp_id,))
        data = cursor.fetchone()

        if data:
            column_names = [description[0] for description in cursor.description]
            return data, column_names

        return None, None

    except sqlite3.Error as err:
        print(f"SQLite error: {err}")
        return None, None
    finally:
        if 'connection' in locals():
            connection.close()


def transform_data(data: tuple, columns: list) -> Optional[Dict]:
    """Transform the fetched data into a structured format."""
    if not data:
        return None

    # Create indices dictionary for easier column access
    col_idx = {col: idx for idx, col in enumerate(columns)}

    # Extract course info
    course_info = {
        'comp': data[col_idx['comp']],
        'hippodrome': data[col_idx['hippodrome']],
        'jour': data[col_idx['jour']],
        'meteo': data[col_idx['meteo']],
        'dist': convert_decimal(data[col_idx['dist']]),
        'corde': data[col_idx['corde']],
        'natpis': data[col_idx['natpis']],
        'pistegp': data[col_idx['pistegp']],
        'typec': data[col_idx['typec']],
        'temperature': convert_decimal(data[col_idx['temperature']]),
        'forceVent': convert_decimal(data[col_idx['forceVent']]),
        'directionVent': data[col_idx['directionVent']],
        'nebulosite': data[col_idx['nebulosite']]
    }

    # Parse participants JSON
    try:
        participants = json.loads(data[col_idx['participants']])
    except (json.JSONDecodeError, TypeError):
        participants = []

    # Transform participants data
    transformed_participants = []
    for p in participants:
        participant_info = {
            'idche': convert_decimal(p.get('idche', 0)),
            'cheval': p.get('cheval', ''),
            'numero': int(p.get('numero', 0)),
            'age': int(p.get('age', 0)),
            'musiqueche': p.get('musiqueche', ''),
            'idJockey': convert_decimal(p.get('idJockey', 0)),
            'musiquejoc': p.get('musiquejoc', ''),
            'idEntraineur': convert_decimal(p.get('idEntraineur', 0)),
            'cotedirect': convert_decimal(p.get('cotedirect', 0))
        }
        transformed_participants.append(participant_info)

    return {
        'course_info': course_info,
        'participants': transformed_participants
    }


def main(comp_id: int) -> Optional[str]:
    """Main function to fetch and transform race data."""
    # Load configuration
    config = setup_environment()

    # Get database path
    config = setup_environment()
    # Get active database path
    db_path = get_database_path(config)
    data, columns = fetch_race_data(str(db_path), comp_id)
    if data and columns:
        course_data = transform_data(data, columns)
        if course_data:
            return json.dumps(course_data)

    return None


if __name__ == "__main__":

    if len(sys.argv) > 1:
        try:
            comp_id = int(sys.argv[1])
            result = main(comp_id)
            if result:
                print(result)
            else:
                print(f"No data found for competition ID: {comp_id}")
        except ValueError:
            print("Error: Competition ID must be a number")
    else:
        print("Usage: python format_coursedata.py <comp_id>")