import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import json

from core.format_coursedata import main as fetch_next_race  # Import the main function from fetch_data.py


def connect_to_db(db_path):
    """Connect to the SQLite database."""
    return sqlite3.connect(db_path)


def fetch_combined_data(conn):
    """Fetch combined race and historical performance data from the database."""
    query = """
    SELECT 
        c.comp, 
        r.ordre_arrivee,
        r.idche,
        c.jour,
        c.participants,
        c.typec,
        c.natpis
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
                'idche': horse.get('idche'),
                'narrivee': horse.get('narrivee'),
                'age': next((p.get('age') for p in row['participants'] if p.get('idche') == horse.get('idche')), None),
                'typec': row['typec'],
                'natpis': row['natpis']
            })

    df_results = pd.DataFrame(results)
    df_results['position'] = pd.to_numeric(df_results['narrivee'], errors='coerce').fillna(0)
    return df_results


def aggregate_results(df_results):
    """Aggregate results to get average position and count, grouped by horse and race type."""
    return df_results.groupby(['idche', 'typec','natpis']).agg(
        raw_avg_position=('position', 'mean'),
        count=('position', 'count')
    ).reset_index()


def prepare_historical_data(df_results):
    """Prepare historical data for training the model."""
    historical_data = aggregate_results(df_results)
    return historical_data


def encode_categorical_features(df):
    """Encode categorical features."""
    df_encoded = pd.get_dummies(df, columns=['typec', 'natpis'], drop_first=True)
    return df_encoded


def prepare_next_race_data(next_race_data, historical_data):
    """Prepare the next race data for prediction."""
    df_next_race = pd.DataFrame(next_race_data['participants'])

    # Merge historical data to get features
    df_next_race = df_next_race.merge(historical_data, on='idche', how='left', suffixes=('', '_historical'))

    # Add additional features from next_race_data if needed
    df_next_race['natpis'] = next_race_data['course_info'].get('natpis', None)
    df_next_race['typec'] = next_race_data['course_info'].get('typec', None)


    # Fill missing values if necessary
    df_next_race.fillna(0, inplace=True)

    return df_next_race


def main(comp_to_predict):
    # Connect to the database
    conn = connect_to_db('data/hippique.db')

    # Fetch combined data
    data = fetch_combined_data(conn)
    conn.close()

    # Process results
    df_results = process_results(data)

    # Prepare historical data for training
    historical_data = prepare_historical_data(df_results)

    # Data for the next race
    next_race_data = json.loads(fetch_next_race(comp_to_predict))

    # Prepare the next race data
    df_next_race = prepare_next_race_data(next_race_data, historical_data)

    df_next_race = encode_categorical_features(df_next_race)

        # Prepare training data
    X = historical_data.drop(columns=['idche', 'raw_avg_position', 'count'])
    y = historical_data['idche']
    X = X.values.reshape(-1, 1)

# Modélisation
    model = LinearRegression()
    model.fit(X, y)

# Prédiction
    future_courses = np.array([[x] for x in range(len(data), len(data) + 5)])  # 5 prochaines courses
    predictions = model.predict(future_courses)

# Visualisation
    plt.scatter(X, y, color='blue')
    plt.plot(X, model.predict(X), color='red')
    plt.scatter(future_courses, predictions, color='green', marker='x')
    plt.title('Performances des Chevaux')
    plt.xlabel('Course')
    plt.ylabel('Position')
    plt.show()