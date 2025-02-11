import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate
from tensorflow.keras.models import Model


def create_rf_model():
    """Create a Random Forest model with tuned hyperparameters."""
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )


def create_lstm_model(sequence_length, seq_feature_dim, static_feature_dim):
    """Create an LSTM model for sequential race predictions.

    Args:
        sequence_length: Number of time steps in sequence
        seq_feature_dim: Number of features in each sequence step
        static_feature_dim: Number of static features
    """
    print(f"Creating LSTM model with shapes:")
    print(f"Sequence input shape: ({sequence_length}, {seq_feature_dim})")
    print(f"Static input shape: ({static_feature_dim},)")

    # Sequential input branch
    seq_input = Input(shape=(sequence_length, seq_feature_dim), name='sequence_input')
    lstm_out = LSTM(64, return_sequences=False)(seq_input)

    # Static features branch
    static_input = Input(shape=(static_feature_dim,), name='static_input')

    # Combine both types of features
    combined = Concatenate()([lstm_out, static_input])

    # Dense layers for final prediction
    dense1 = Dense(64, activation='relu')(combined)
    dense2 = Dense(32, activation='relu')(dense1)
    output = Dense(1)(dense2)

    model = Model(inputs=[seq_input, static_input], outputs=output)
    model.compile(optimizer='adam', loss='huber', metrics=['mae'])

    # Print model summary for debugging
    model.summary()

    return model


def create_hybrid_model(sequence_length, seq_feature_dim, static_feature_dim):
    """Create a hybrid model combining RF and LSTM."""
    rf_model = create_rf_model()
    lstm_model = create_lstm_model(sequence_length, seq_feature_dim, static_feature_dim)

    return {
        'rf': rf_model,
        'lstm': lstm_model
    }