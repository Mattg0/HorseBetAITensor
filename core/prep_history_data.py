import os
import pandas as pd
import json
import sqlite3
import hashlib
from env_setup import setup_environment
from concurrent.futures import ThreadPoolExecutor
from functools import partial


def connect_to_db(db_path):
    """Connect to the SQLite database with optimized settings."""
    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA journal_mode = WAL')  # Write-Ahead Logging
    conn.execute('PRAGMA cache_size = -2000')  # 2MB cache
    conn.execute('PRAGMA synchronous = NORMAL')
    return conn


def fetch_combined_data(conn):
    """Fetch combined race and historical performance data efficiently."""
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
        c.dist,
        c.temperature
    FROM 
        Course c
    JOIN 
        Resultats r ON c.comp = r.comp 
    """
    # Use chunked reading for large datasets
    chunks = []
    for chunk in pd.read_sql_query(query, conn, chunksize=10000):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def safe_json_loads(x):
    """Safely load JSON string, handling potential errors."""
    try:
        return json.loads(str(x).replace("'", '"'))
    except (json.JSONDecodeError, AttributeError, TypeError):
        return []


def process_single_row(row_data):
    """Process a single row of race data."""
    row, participants = row_data
    results = []

    def safe_float_convert(value, default=0.00):
        try:
            return float(value) if value not in ['N/A', '', None] else default
        except (ValueError, TypeError):
            return default

    ordre_arrivee = safe_json_loads(row['ordre_arrivee'])
    participants_data = safe_json_loads(row['participants'])

    if not ordre_arrivee or not participants_data:
        return []

    for horse in ordre_arrivee:
        participant = next((p for p in participants_data if p.get('idche') == horse.get('idche')), {})

        results.append({
            'comp': row['comp'],
            'jour': row['jour'],
            'idche': horse.get('idche'),
            'narrivee': horse.get('narrivee'),
            'age': participant.get('age'),
            'musiqueche': participant.get('musiqueche'),
            'musiquejoc': participant.get('musiquejoc'),
            'idJockey': participant.get('idJockey'),
            'idEntraineur': participant.get('idEntraineur'),
            'cotedirect': safe_float_convert(participant.get('cotedirect')),
            'typec': row['typec'],
            'natpis': row['natpis'],
            'dist': safe_float_convert(row['dist']),
            'corde': row['corde'],
            'meteo': row['meteo'],
            'temperature': safe_float_convert(row['temperature'])
        })

    return results


def process_results(data):
    """Process results using parallel processing."""
    # Pre-process JSON data
    print("Pre-processing JSON data...")
    data['participants'] = data['participants'].astype(str).apply(safe_json_loads)
    data['ordre_arrivee'] = data['ordre_arrivee'].astype(str).apply(safe_json_loads)

    # Prepare data for parallel processing
    print("Preparing data for parallel processing...")
    row_data = [(row, row['participants']) for _, row in data.iterrows()]

    # Process in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(process_single_row, row_data))

    # Flatten results
    flattened_results = [item for sublist in results for item in sublist]

    # Convert to DataFrame
    df_results = pd.DataFrame(flattened_results)
    df_results['position'] = pd.to_numeric(df_results['narrivee'], errors='coerce').fillna(0)

    # Optimize memory usage
    df_results = df_results.astype({
        'comp': 'int32',
        'idche': 'int32',
        'age': 'float32',
        'idJockey': 'int32',
        'idEntraineur': 'int32',
        'cotedirect': 'float32',
        'dist': 'float32',
        'temperature': 'float32',
        'position': 'float32'
    })

    return df_results.dropna(subset=['age', 'idJockey'])


def assign_value_to_combinations(df):
    """Optimize hash computation for categorical variables."""
    categorical_cols = ['natpis', 'typec', 'meteo', 'corde']

    # Pre-compute unique values and their hashes
    hash_maps = {}
    for col in categorical_cols:
        unique_vals = df[col].unique()
        hash_maps[col] = {
            val: int(hashlib.md5(str(val).encode()).hexdigest(), 16) % (10 ** 8)
            for val in unique_vals
        }
        df[col] = df[col].map(hash_maps[col])

    df['jour'] = pd.to_datetime(df['jour']).astype(int) // 10 ** 9
    return df


def main():
    config = setup_environment()
    root_dir = config['rootdir']
    os.chdir(root_dir)

    # Cache file path
    cache_dir = os.path.join(root_dir, config['paths']['cache'])
    cache_file = os.path.join(cache_dir, 'processed_results.parquet')

    # Check if cached results exist
    if os.path.exists(cache_file):
        return pd.read_parquet(cache_file)

    # Process data if cache doesn't exist
    db_config = next((db for db in config['databases'] if db['name'] == '2years'), None)
    if not db_config:
        raise ValueError("Lite database configuration not found")

    db_path = os.path.join(root_dir, db_config['path'])
    db_name = 'data/hippique.db'
    conn = connect_to_db(db_path)

    data = fetch_combined_data(conn)
    conn.close()

    df_results = process_results(data)
    df_results = assign_value_to_combinations(df_results)

    # Save to cache
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    df_results.to_parquet(cache_file)

    return df_results