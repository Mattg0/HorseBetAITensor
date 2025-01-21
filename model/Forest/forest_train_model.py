import sys
sys.path.append('../../')
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import datetime
from env_setup import setup_environment
from core.prep_history_data import main as get_historical_races


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

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
    return model

def save_model_and_scaler(model, scaler, label_encoder_idche, label_encoder_idJockey, config):
    """Save the model and scaler to disk using paths from the config."""
    # Get paths from the configuration
    model_path = config['model'][1]['filepath']+config['model'][1]['racemodelKERAS']  # Adjust index if necessary
    scaler_path = config['model'][1]['filepath']+config['model'][1]['scaler']  # Adjust index if necessary
    label_encoder_idche_path = config['model'][1]['filepath']+config['model'][1]['label_encoder_idche']  # Adjust index if necessary
    label_encoder_idJockey_path = config['model'][1]['filepath']+config['model'][1]['label_encoder_idJockey']  # Adjust index if necessary

    # Save the model and scaler to disk
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder_idche, label_encoder_idche_path)
    joblib.dump(label_encoder_idJockey, label_encoder_idJockey_path)

def main():
    config = setup_environment()
    # get_data
    df_results = get_historical_races()

    # Prepare features and target variable
    X = df_results[['idche', 'jour', 'idJockey', 'age', 'typec', 'natpis', 'meteo', 'dist', 'corde','cotedirect']]
    y = df_results['position'].astype(int)  # Ensure target variable is integer

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
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1,callbacks=[tensorboard_callback])

    # Save the model, scaler, and label encoders
    save_model_and_scaler(model, scaler, le_idche, le_idJockey,config)

    # Evaluate the model
    loss, mae = model.evaluate(X_test_scaled, y_test)
    print(f'Test Loss: {loss}, Test MAE: {mae}')

if __name__ == "__main__":
    main()