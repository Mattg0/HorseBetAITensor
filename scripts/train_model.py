import pandas as pd
import json
import sqlite3
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import hashlib
import numpy as np  # Import numpy for numerical checks

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
                    'idche': horse.get('idche'),
                    'narrivee': horse.get('narrivee'),
                    'age': next((p.get('age') for p in row['participants'] if p.get('idche') == horse.get('idche')),
                                None),
                    'idJockey': next(
                        (p.get('idJockey') for p in row['participants'] if p.get('idche') == horse.get('idche')), None),
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



    return df

def encode_targets(y):
    """Encode target variable."""
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(y), label_encoder

def encode_categorical_features(df):
    """Encode categorical features using One-Hot Encoding."""
    df_encoded = pd.get_dummies(df, columns=['typec', 'natpis'], drop_first=True)
    return df_encoded

def build_model(input_shape):
    """Build and compile the neural network model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Assuming regression on finishing position
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='mean_squared_error', metrics=['mae'])
    return model

def save_model_and_scaler(model, scaler, label_encoder_idche, label_encoder_idJockey):
    """Save the model and scaler to disk."""
    model.save('model/race_model.keras')
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(label_encoder_idche, 'model/label_encoder_idche.pkl')
    joblib.dump(label_encoder_idJockey, 'model/label_encoder_idJockey.pkl')

def main():
    # Connect to the database
    conn = connect_to_db('data/hippique.db')

    # Fetch and process the combined data
    data = fetch_combined_data(conn)
    df_results = process_results(data)
    conn.close()

    # Check for NaN values in df_results
    print("Checking for NaN values in df_results:")
   # print(df_results.isnull().sum())

    # Assign unique integer values to combinations of natpis, typec, and meteo
    df_results = assign_value_to_combinations(df_results)

    # Prepare features and target variable
    X = df_results[['idche', 'idJockey', 'age', 'typec','natpis','meteo','dist','corde']]
    y = df_results['position'].astype(int)  # Ensure target variable is integer

    # Check for NaN or infinite values in X and y
    print("Checking for NaN values in X:")
    print(X.isnull().sum())
    print("Checking for infinite values in X:")
    print(np.isinf(X).sum())
    print("Checking for NaN values in y:")
    print(y.isnull().sum())
    print("Checking for infinite values in y:")
    print(np.isinf(y).sum())

    # Proceed only if there are no NaN values
    if X.isnull().any().any() or y.isnull().any():
        print("Data contains NaN values. Please address this before training the model.")
        return

    # Encode categorical variables
    le_idche = LabelEncoder()
    le_idJockey = LabelEncoder()

    X['idche'] = le_idche.fit_transform(X['idche'])
    X['idJockey'] = le_idJockey.fit_transform(X['idJockey'])

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build and compile the model
    model = build_model(X_train_scaled.shape[1])

    # Train the model
    print('epoch start')
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=16, validation_split=0.2, verbose=1)
    print('epoch end'
          '')
    # Save the model, scaler, and label encoders
    save_model_and_scaler(model, scaler, le_idche, le_idJockey)

    # Evaluate the model
    loss, mae = model.evaluate(X_test_scaled, y_test)
    print(f'Test Loss: {loss}, Test MAE: {mae}')

if __name__ == "__main__":
    main()