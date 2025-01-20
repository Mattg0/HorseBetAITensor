import pandas as pd
import json
import sqlite3
import tensorflow as tf
import joblib  # For loading the scaler and label encoder
from core.format_coursedata import main as fetch_next_race
import hashlib
import numpy as np

def connect_to_db(db_path):
    """Connect to the SQLite database."""
    return sqlite3.connect(db_path)

def load_model_and_scaler():
    """Load the model and scaler from disk."""
    model = tf.keras.models.load_model('model/race_model.keras')
    scaler = joblib.load('model/scaler.pkl')
    label_encoder_idche = joblib.load('model/label_encoder_idche.pkl')
    label_encoder_idJockey = joblib.load('model/label_encoder_idJockey.pkl')
    return model, scaler, label_encoder_idche, label_encoder_idJockey

def assign_value_to_combinations(df):
    """Assign a unique integer value to each combination of natpis, typec, and meteo."""
    """Assign a unique integer value to each combination of natpis, typec, and meteo."""
    df['natpis'] = df['natpis'].apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % (10 ** 8))
    df['typec'] = df['typec'].apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % (10 ** 8))
    df['meteo'] = df['meteo'].apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % (10 ** 8))
    df['corde'] = df['corde'].apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % (10 ** 8))

    return df

def main(comp_to_predict):
    # Fetch the next race data
    next_race_data = json.loads(fetch_next_race(comp_to_predict))

    # Prepare the next race data
    df_next_race = pd.DataFrame(next_race_data['participants'])
    df_next_race['natpis'] = next_race_data['course_info'].get('natpis', None)
    df_next_race['typec'] = next_race_data['course_info'].get('typec', None)
    df_next_race['meteo'] = next_race_data['course_info'].get('meteo', None)
    df_next_race['dist'] = next_race_data['course_info'].get('dist', None)
    df_next_race['corde'] = next_race_data['course_info'].get('corde', None)


    # Load the model and scaler
    model, scaler, label_encoder_idche, label_encoder_idJockey = load_model_and_scaler()
    print("Loaded existing model and scaler.")

    # Print model summary for debugging
    model.summary()

    # Assign race profiles to the next race data
    df_next_race = assign_value_to_combinations(df_next_race)

    # Encode categorical variables
#    df_next_race['idche'] = label_encoder_idche.transform(df_next_race['idche'])
 #   df_next_race['idJockey'] = label_encoder_idJockey.transform(df_next_race['idJockey'])

    # Check prepared input data
    print("Prepared input data:")
 #   print(df_next_race[['idche', 'idJockey', 'age']])

    # Prepare the input features for the model
    X_next_race = df_next_race[['idche', 'idJockey', 'age', 'dist','natpis','typec','corde','meteo']]

    # Scale the next race features
    X_next_race_scaled = scaler.transform(X_next_race)

    # Check scaled features
    print("Scaled features:")
    print(X_next_race_scaled)

    # Make predictions
    predictions = model.predict(X_next_race_scaled)

    # Check predictions
    print("Predictions:")
    print(predictions)

    # Check for NaN predictions
    if np.any(np.isnan(predictions)):
        print("Predictions contain NaN values. Check model inputs and architecture.")
        return

    # Add predictions to df_next_race
    df_next_race['predicted_scores'] = predictions.flatten()

    # Check predicted scores
    print("Predicted scores:")
    print(df_next_race['predicted_scores'])

    # Sort by predicted scores to determine positions
    df_next_race['predicted_ordre_arrivee'] = df_next_race['predicted_scores'].rank(method='min', ascending=False)

    # Sort the DataFrame by predicted order
    result = df_next_race.sort_values(by='predicted_ordre_arrivee')

    # Construct output with each horse's idche and predicted order
    result = result[['idche', 'cheval', 'predicted_ordre_arrivee']]
    result['predicted_ordre_arrivee'] = result['predicted_ordre_arrivee'].astype(int)  # Ensure it's an integer#
    result = result.merge(pd.DataFrame(next_race_data['participants']), on='idche', how='left',
                          suffixes=('', '_participants'))

    result = df_next_race.sort_values(by='predicted_scores', ascending=False).head(5)

    # Join results with participants on 'idche'
    result = result.merge(pd.DataFrame(next_race_data['participants']), on='idche', how='left',
                          suffixes=('', '_participants'))

    # Sort by predicted score and assign a unique order
    result['unique_order'] = range(1, len(result) + 1)

    # Select only desired columns
    predict_arriv = result['numero'].astype(str).str.cat(sep='-')

    # Display the result
    print(predict_arriv)
    return predict_arriv

if __name__ == "__main__":
    main('1552621')  # Replace with the actual competition ID you want to predict