import pandas as pd
import json
import sqlite3
import re
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
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
                'comp': row['comp'],
                'idche': horse.get('idche'),
                'narrivee': horse.get('narrivee'),
                'age': next((p.get('age') for p in row['participants'] if p.get('idche') == horse.get('idche')), None),
                'idJockey': next((p.get('idJockey') for p in row['participants'] if p.get('idche') == horse.get('idche')), None),
                'typec': row['typec'],
                'natpis': row['natpis']
            })

    df_results = pd.DataFrame(results)
    df_results['position'] = pd.to_numeric(df_results['narrivee'], errors='coerce').fillna(0)
    return df_results


def aggregate_results(df_results, next_race_data):
    """Aggregate results to get average position and count, grouped by horse and race type."""
    aggregated_results = []

    for index in range(len(next_race_data)):
        target_idche = next_race_data['idche'].iloc[index]
        target_typec = next_race_data['typec'].iloc[index]
        target_natpis = next_race_data['natpis'].iloc[index]
        target_idJockey = next_race_data['idJockey'].iloc[index]

        filtered_df = df_results[
            (df_results['idche'] == target_idche) &
            (df_results['typec'] == target_typec) &
            (df_results['natpis'] == target_natpis) &
            (df_results['idJockey'] == target_idJockey)
        ]

        count_df = df_results.groupby(['idche', 'typec', 'natpis', 'idJockey']).size().reset_index(name='count')
        df_results_with_count = df_results.merge(count_df, on=['idche', 'typec', 'natpis', 'idJockey'], how='left')

        weighted_result = df_results_with_count.groupby(['idche', 'typec', 'natpis', 'idJockey']).agg(
            weighted_position=('position', lambda x: (x * df_results_with_count.loc[x.index, 'count']).sum() /
                                                     df_results_with_count.loc[x.index, 'count'].sum()),
            count=('position', 'size')
        ).reset_index()

        aggregated_results.append(weighted_result)

    final_results = pd.concat(aggregated_results, ignore_index=True)
    final_results = df_results.merge(final_results, on='idche', how='left')

    return final_results


def encode_targets(y):
    """Encode target variable."""
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(y), label_encoder


def encode_categorical_features(df):
    """Encode categorical features."""
    df_encoded = pd.get_dummies(df, columns=['typec_x', 'natpis_x'], drop_first=True)
    return df_encoded


def build_model(input_shape, num_classes):
    """Build and compile the neural network model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def prepare_next_race_data(next_race_data):
    """Prepare the next race data for prediction."""
    df_next_race = pd.DataFrame(next_race_data['participants'])
    df_next_race['natpis'] = next_race_data['course_info'].get('natpis', None)
    df_next_race['typec'] = next_race_data['course_info'].get('typec', None)
    df_next_race.fillna(0, inplace=True)
    return df_next_race


def main(comp_to_predict):
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    physical_devices = tf.config.list_physical_devices()
    for device in physical_devices:
        print(device)

    # Connect to the database
    conn = connect_to_db('data/hippique.db')

    # Fetch combined data
    data = fetch_combined_data(conn)
    conn.close()

    # Process results
    df_previous_races = process_results(data)

    # Data for the next race
    next_race_data = json.loads(fetch_next_race(comp_to_predict))

    # Prepare the next race data
    df_next_race = prepare_next_race_data(next_race_data)
    df_prediction_data = aggregate_results(df_previous_races, df_next_race)

    if not df_next_race.empty:
        # Encode categorical features
        df_prediction_data = encode_categorical_features(df_prediction_data)
        df_prediction_data = df_prediction_data.drop(columns=['typec_x_Haies', 'typec_x_Monté', 'typec_x_Plat', 'typec_x_Steeple-chase', 'natpis_x_N/A', 'natpis_x_PH', 'natpis_x_PS', 'natpis_x_PSF'])
        df_prediction_data = df_prediction_data.drop_duplicates(keep='last')

        # Prepare training data
        X = df_prediction_data.drop(columns=['idche', 'weighted_position', 'count'])
        y = df_prediction_data['idche']

        # Encode targets
        y_encoded, label_encoder = encode_targets(y)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # Ensure all columns are numeric
        X_train = X_train.apply(pd.to_numeric, errors='coerce')
        X_test = X_test.apply(pd.to_numeric, errors='coerce')

        # Drop any rows with NaN values
        X_train.dropna(inplace=True)
        y_train = y_train[X_train.index]  # Align y_train with X_train

        print(f"Shape of X_train: {X_train.shape}")
        print(f"Shape of y_train: {y_train.shape}")

        # Scale the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create and train the model
        model = build_model(X_train.shape[1], len(label_encoder.classes_))
        model.fit(X_train, y_train, epochs=1, batch_size=128, verbose=1)

        # Prepare features for prediction
        X_next_race = df_prediction_data.drop(columns=['idche', 'typec_y', 'natpis_y', 'idJockey_y'])

        # Ensure X_next_race has the same columns as X_train
        X_next_race = X_next_race[X_train.columns]

        # Scale the next race features
        X_next_race = scaler.transform(X_next_race)

        # Predict the finishing order for the next race
        predictions = model.predict(X_next_race)

        # Add predictions to df_next_race
        df_prediction_data['predicted_scores'] = predictions.max(axis=1)
        predicted_classes = tf.argmax(predictions, axis=1).numpy()
        df_prediction_data['predicted_ordre_arrivee'] = predicted_classes

        # Sort by predicted score and select the top 5 participants
        result = df_next_race.sort_values(by='predicted_scores', ascending=False).head(5)

        # Join results with participants on 'idche'
        result = result.merge(pd.DataFrame(next_race_data['participants']), on='idche', how='left', suffixes=('', '_participants'))

        # Sort by predicted score and assign a unique order
        result['unique_order'] = range(1, len(result) + 1)

        # Select only desired columns
        predict_arriv = result['numero_x'].astype(str).str.cat(sep='-')

        # Display the result
        print(predict_arriv)
        return predict_arriv

    else:
        print("The DataFrame df_next_race is empty after preparation.")


if __name__ == "__main__":
    main('931496')