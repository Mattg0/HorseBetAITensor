import joblib
import pandas as pd
import tensorflow as tf


def load_model_and_scaler():
    model = tf.keras.models.load_model('model/race_model.keras')
    scaler = joblib.load('model/scaler.pkl')
    label_encoder = joblib.load('model/label_encoder.pkl')

    return model, scaler, label_encoder


def display_feature_columns():
    # Load your training data (or keep the columns from training)
    df_train = pd.read_csv('path_to_your_training_data.csv')  # Adjust this as necessary
    feature_columns = df_train.drop(
        columns=['target']).columns.tolist()  # Replace 'target' with your target column name

    print("Feature columns used in the model:")
    print(feature_columns)


# Example usage
model, scaler, label_encoder = load_model_and_scaler()
display_feature_columns()