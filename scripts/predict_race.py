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
    df['natpis'] = df['natpis'].apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % (10 ** 8))
    df['typec'] = df['typec'].apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % (10 ** 8))
    df['meteo'] = df['meteo'].apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % (10 ** 8))
    df['corde'] = df['corde'].apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % (10 ** 8))
    #df['jour'] = pd.to_datetime(df['jour']).astype(int) // 10 ** 9
    return df

# Encode categorical variables with handling for unseen labels
def safe_transform(encoder, values):
        transformed_values = []
        for val in values:
            try:
                transformed_values.append(encoder.transform([val])[0])
            except ValueError:
                transformed_values.append(-1)  # Fallback for unseen labels
        return np.array(transformed_values)


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

  #   df_next_race['idche'] = safe_transform(label_encoder_idche, df_next_race['idche'])
   #  df_next_race['idJockey'] = safe_transform(label_encoder_idJockey, df_next_race['idJockey'])

    # Prepare the input features for the model
     feature_columns = ['idche', 'idJockey', 'age', 'typec', 'natpis', 'meteo', 'dist', 'corde']

    # Check if all required columns exist
     missing_cols = set(feature_columns) - set(df_next_race.columns)
     if missing_cols:
         print(f"Missing columns in input data: {missing_cols}")
         return

     X_next_race = df_next_race[feature_columns]

    # Scale the next race features
     X_next_race_scaled = scaler.transform(X_next_race)

    # No need to reshape for dense models
    # X_next_race_reshaped = X_next_race_scaled.reshape((X_next_race_scaled.shape[0], 1, X_next_race_scaled.shape[1]))

    # Make predictions
     predictions = model.predict(X_next_race_scaled)  # Input shape should now be (16, 8)

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
     result['predicted_ordre_arrivee'] = result['predicted_ordre_arrivee'].astype(int)  # Ensure it's an integer

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
    main('1552621')