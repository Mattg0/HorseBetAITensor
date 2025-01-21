import os
import pandas as pd
import json
import sqlite3
import hashlib
from env_setup import setup_environment

config = setup_environment()
# Change the current working directory to the root directory
root_dir = config['rootdir']
os.chdir(root_dir)

def connect_to_db(db_path):
    """Connect to the SQLite database."""
    return sqlite3.connect(db_path)

def fetch_combined_data(conn):
    """Fetch combined race and historical performance data from the database."""
    query = """
    SELECT 
        c.comp, 
        r.ordre_arrivee,
        c.jour,
        c.participants,
        c.typec,
        c.natpis,
        c.meteo,
        c.corde,
        c.dist
    FROM 
        Course c
    JOIN 
        Resultats r ON c.comp = r.comp 
    """
    return pd.read_sql_query(query, conn)

def process_results(data):
    """Process the results and extract relevant information."""

    def safe_json_loads(x):
        try:
            x = x.replace("'", '"')  # Replace single quotes with double quotes
            return json.loads(x)
        except json.JSONDecodeError:
            return []  # Return an empty list or handle as needed

    data['participants'] = data['participants'].astype(str).apply(safe_json_loads)
    data['ordre_arrivee'] = data['ordre_arrivee'].astype(str).apply(safe_json_loads)

    results = []
    for index, row in data.iterrows():
        for horse in row['ordre_arrivee']:
            results.append({
                'comp': row['comp'],
                'jour': row['jour'],
                'idche': horse.get('idche'),
                'narrivee': horse.get('narrivee'),
                'age': next((p.get('age') for p in row['participants'] if p.get('idche') == horse.get('idche')), None),
                'idJockey': next((p.get('idJockey') for p in row['participants'] if p.get('idche') == horse.get('idche')), None),
                'cotedirect': float(next((p.get('cotedirect') for p in row['participants'] if p.get('idche') == horse.get('idche')),0.00) or 0.00),
                'typec': row['typec'],
                'natpis': row['natpis'],
                'dist': row['dist'],
                'corde': row['corde'],
                'meteo': row['meteo']
            })

    df_results = pd.DataFrame(results)
    df_results['position'] = pd.to_numeric(df_results['narrivee'], errors='coerce').fillna(0)

    # Drop rows where 'age' or 'idJockey' is NaN
    df_results = df_results.dropna(subset=['age', 'idJockey'])

    return df_results

def assign_value_to_combinations(df):
    """Assign a unique integer value to each combination of natpis, typec, and meteo."""
    df['natpis'] = df['natpis'].apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % (10 ** 8))
    df['typec'] = df['typec'].apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % (10 ** 8))
    df['meteo'] = df['meteo'].apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % (10 ** 8))
    df['corde'] = df['corde'].apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % (10 ** 8))
    df['jour'] = pd.to_datetime(df['jour']).astype(int) // 10**9  # Convert to UNIX timestamp
    return df

def main():
    # Load configuration
    config = setup_environment()
    # Change the current working directory to the root directory
    root_dir = config['rootdir']
    os.chdir(root_dir)

    # Get the database path from the config
    db_name = config['databases'][0]['relative_path']

    # Connect to DB
    conn= connect_to_db(db_name)
    # Fetch and process the combined data

    data = fetch_combined_data(conn)
    df_results = process_results(data)
    conn.close()

    # Check for NaN values in df_results
    print("Checking for NaN values in df_results:")
    print(df_results.isnull().sum())

    # Assign unique integer values to combinations of natpis, typec, and meteo
    df_results = assign_value_to_combinations(df_results)

    return df_results

