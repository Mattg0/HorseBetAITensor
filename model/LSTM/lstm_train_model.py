import pandas as pd
import json
import sqlite3
import sys
sys.path.append('../..')
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import hashlib
import numpy as np
import os
from core.prep_history_data import main as get_historical_races


def encode_targets(y):
    """Encode target variable."""
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(y), label_encoder

def build_lstm_model(input_shape):
    """Build and compile the LSTM neural network model."""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Assuming regression on finishing position
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='mean_squared_error', metrics=['mae'])
    return model

def save_model_and_scaler(model, scaler, label_encoder_idche, label_encoder_idJockey):
    """Save the model and scaler to disk."""
    os.makedirs('model', exist_ok=True)  # Create the directory if it doesn't exist
    model.save('model/lstm_race_model.keras')
    joblib.dump(scaler, 'model/lstm_scaler.pkl')
    joblib.dump(label_encoder_idche, 'model/lstm_label_encoder_idche.pkl')
    joblib.dump(label_encoder_idJockey, 'model/lstm_label_encoder_idJockey.pkl')

def main():

    # Assign unique integer values to combinations
    df_results = get_historical_races()

    # Prepare features and target variable
    X = df_results[['idche', 'jour', 'idJockey', 'age', 'typec', 'natpis', 'meteo', 'dist', 'corde','cotedirect']]
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

    X.loc[:, 'idche'] = le_idche.fit_transform(X['idche'])
    X.loc[:, 'idJockey'] = le_idJockey.fit_transform(X['idJockey'])

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape the data for LSTM
    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # Build and compile the LSTM model
    model = build_lstm_model((X_train_reshaped.shape[1], X_train_reshaped.shape[2]))

    # Train the model
    print('Training model...')
    model.fit(X_train_reshaped, y_train, epochs=10, batch_size=16, validation_split=0.2, verbose=1)

    # Save the model, scaler, and label encoders
    save_model_and_scaler(model, scaler, le_idche, le_idJockey)

    # Evaluate the model
    loss, mae = model.evaluate(X_test_reshaped, y_test)
    print(f'Test Loss: {loss}, Test MAE: {mae}')

if __name__ == "__main__":
    main()