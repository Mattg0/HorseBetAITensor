import pandas as pd
import json
import sqlite3
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from core.format_coursedata import main as fetch_next_race  # Importer la fonction main depuis fetch_data.py


def connect_to_db(db_path):
    """Connect to the SQLite database."""
    return sqlite3.connect(db_path)


def fetch_data(conn):
    """Fetch race data from the database."""
    query = """
    SELECT c.*, r.ordre_arrivee, c.meteo, c.dist, c.corde, c.natpis
    FROM Course c
    JOIN Resultats r ON c.comp = r.comp 
    WHERE c.corde IS NOT NULL OR c.corde != ''
    """
    return pd.read_sql_query(query, conn)


def process_results(data):
    """Process the results and extract relevant information."""
    data['participants'] = data['participants'].apply(json.loads)
    data['ordre_arrivee'] = data['ordre_arrivee'].apply(json.loads)

    results = []
    for index, row in data.iterrows():
        for horse in row['ordre_arrivee']:
            results.append({
                'idche': horse['idche'],
                'narrivee': horse['narrivee'],
            })

    df_results = pd.DataFrame(results)
    df_results['position'] = pd.to_numeric(df_results['narrivee'], errors='coerce').fillna(0)
    return df_results


def aggregate_results(df_results):
    """Aggregate results to get average position and count."""
    return df_results.groupby('idche').agg(
        avg_position=('position', 'mean'),
        count=('position', 'count')
    ).reset_index()


def prepare_next_race_data(next_race_data, df_stats):
    """Prepare the next race data for prediction."""
    # Convertir les participants en DataFrame
    df_next_race = pd.DataFrame(next_race_data['participants'])

    # Fusionner avec les statistiques des chevaux
    df_next_race = df_next_race.merge(df_stats, on='idche', how='left')

    # Remplir les valeurs manquantes
    df_next_race.loc[:, 'avg_position'] = df_next_race['avg_position'].fillna(df_next_race['avg_position'].mean())
    df_next_race.loc[:, 'count'] = df_next_race['count'].fillna(0)

    # Extraire les informations de course_info
    course_info = next_race_data['course_info']

    # Ajouter les caractéristiques de course_info au DataFrame des participants
    for key in ['hippo', 'meteo', 'dist', 'corde', 'natpis', 'pistegp', 'temperature', 'forceVent', 'directionVent',
                'nebulosite']:
        df_next_race[key] = course_info.get(key)
    return df_next_race


def encode_targets(y):
    """Encode target variable."""
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(y), label_encoder


def encode_categorical_features(df):
    """Encode categorical features."""
    df_encoded = pd.get_dummies(df, columns=['meteo', 'natpis'], drop_first=True)
    return df_encoded


def build_model(input_shape, num_classes):
    """Build and compile the neural network model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),  # Utilisation de Input pour définir la forme d'entrée
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main(testcomp):
    # Connexion à la base de données
    conn = connect_to_db('data/hippique.db')

    # Récupération des données
    data = fetch_data(conn)
    conn.close()

    # Traitement des résultats
    df_results = process_results(data)

    # Agrégation des résultats
    df_stats = aggregate_results(df_results)

    next_race_data = json.loads(fetch_next_race(testcomp))


    # Convertir les participants en DataFrame
    df_participants = pd.DataFrame(next_race_data['participants'])

    # Préparation des données de la prochaine course
    df_next_race = prepare_next_race_data(next_race_data, df_stats)

    # Inclure idjoc dans les caractéristiques
    df_next_race = df_next_race.merge(df_participants[['idche','idJockey']], on='idche', how='left')

    # Préparer les données pour le modèle
    df_next_race = encode_categorical_features(df_next_race)
    X_next_race = df_next_race[
        ['avg_position', 'count', 'dist'] + [col for col in df_next_race.columns if 'meteo_' in col] + [col for
                                                                                                                 col in
                                                                                                                 df_next_race.columns
                                                                                                                 if
                                                                                                                 'natpis_' in col]]
    y_next_race = df_next_race['idche']

    # Encoder les cibles
    y_next_race_encoded, label_encoder = encode_targets(y_next_race)

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_next_race, y_next_race_encoded, test_size=0.2,
                                                        random_state=42)

    # Créer et entraîner le modèle
    model = build_model(X_train.shape[1], len(label_encoder.classes_))
    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1)

    # Prédire l'ordre d'arrivée pour la prochaine course
    predictions = model.predict(X_next_race)

    # Ajouter les prédictions à df_next_race
    df_next_race['predicted_scores'] = predictions.max(axis=1)  # Utiliser le score maximum pour le tri
    predicted_classes = tf.argmax(predictions, axis=1).numpy()
    df_next_race['predicted_ordre_arrivee'] = predicted_classes

    # Joindre les résultats avec les participants sur 'idche'
    result = df_next_race.merge(df_participants, on='idche', how='left')

    # Trier par score prédit et attribuer un ordre d'arrivée unique
    result = result.sort_values(by='predicted_scores', ascending=False)
    result['unique_order'] = range(1, len(result) + 1)

    # Sélectionner uniquement les colonnes souhaitées, y compris idjoc
    result = result[['unique_order', 'numero_x', 'cheval_x','idche']]

    # Afficher le résultat
    print(result)


if __name__ == "__main__":
    main('1553862')