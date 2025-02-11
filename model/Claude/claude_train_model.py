from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from typing import Tuple
from core.prep_history_data import main as get_historical_races
from env_setup import setup_environment, get_model_paths
from features import FeatureEngineering
from models.architectures import create_hybrid_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from utils.cache_manager import CacheManager


class HorseRaceModel:
    def __init__(self, config_path: str = 'config.yaml', model_name: str = 'hybrid', sequence_length: int = 5):
        """Initialize the model with configuration."""
        self.config = setup_environment(config_path)
        self.model_paths = get_model_paths(self.config, model_name)
        self.sequence_length = sequence_length
        self.models = None
        self.feature_engineering = FeatureEngineering()
        self.history = None

        # Initialize cache manager
        cache_dir = Path(self.config['paths']['cache'])
        self.cache_manager = CacheManager(cache_dir)

    def prepare_sequence_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare sequential data for LSTM training."""
        sequences = []
        static_features = []
        targets = []

        # Define sequential and static features
        sequential_features = [
            'position',
            'cotedirect',
            'dist',
            'musique_avg_position',
            'musique_top_3_rate'
        ]

        static_features_list = [
            'age',
            'temperature',
            'natpis',
            'typec',
            'meteo',
            'corde',
            'normalized_odds',
            'musique_fault_prone'
        ]

        print("\nFeature dimensions:")
        print(f"Sequential features: {len(sequential_features)}")
        print(f"Static features: {len(static_features_list)}")

        # Convert features to numeric type first
        for col in sequential_features + static_features_list:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Fill NaN values with median for each column
        for col in sequential_features + static_features_list:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        # Group by horse
        for horse_id in df['idche'].unique():
            horse_data = df[df['idche'] == horse_id].sort_values('jour')

            if len(horse_data) >= self.sequence_length + 1:  # Need at least sequence_length + 1 races
                try:
                    # Get sequential features
                    seq_features = horse_data[sequential_features].values.astype(np.float32)

                    # Get static features from the most recent race
                    static_feat = horse_data[static_features_list].iloc[-1].values.astype(np.float32)

                    # Verify no NaN values
                    if not (np.isnan(seq_features).any() or np.isnan(static_feat).any()):
                        # Create sequences
                        for i in range(len(horse_data) - self.sequence_length):
                            sequences.append(seq_features[i:i + self.sequence_length])
                            static_features.append(static_feat)
                            targets.append(float(seq_features[i + self.sequence_length, 0]))  # position is target
                except (ValueError, TypeError) as e:
                    print(f"Error processing horse {horse_id}: {e}")
                    continue

        if not sequences:
            raise ValueError("No valid sequences could be created. Check data quality.")

        # Convert to numpy arrays with explicit types
        sequences = np.array(sequences, dtype=np.float32)
        static_features = np.array(static_features, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        print("\nData shapes:")
        print(f"Sequences shape: {sequences.shape}")
        print(f"Static features shape: {static_features.shape}")
        print(f"Targets shape: {targets.shape}")

        return sequences, static_features, targets

    def train(self, use_cache: bool = True) -> None:
        """Train the hybrid model using historical race data."""
        print("Loading historical race data...")

        # Try to load processed data from cache
        if use_cache:
            cached_data = self.cache_manager.load('processed_training_data')
            if cached_data is not None:
                df_features = cached_data
            else:
                df_historical = get_historical_races()
                df_features = self.feature_engineering.extract_all_features(df_historical)
                self.cache_manager.save(df_features, 'processed_training_data')
        else:
            df_historical = get_historical_races()
            df_features = self.feature_engineering.extract_all_features(df_historical)

        # Prepare static features for RF
        static_columns = self.feature_engineering.get_feature_columns()
        static_features = df_features[static_columns].astype(float)

        print("\nPreparing sequence data...")
        seq_features, static_seq, targets = self.prepare_sequence_data(df_features)

        if len(seq_features) == 0:
            raise ValueError("No valid sequences found in the data. Check sequence length and data availability.")

        # Create hybrid model
        self.models = create_hybrid_model(
            sequence_length=self.sequence_length,
            seq_feature_dim=seq_features.shape[2],  # Number of features per timestep
            static_feature_dim=static_seq.shape[1]  # Number of static features
        )

        print("\nTraining Random Forest model...")
        self.models['rf'].fit(static_features, df_features['position'])

        print("\nTraining LSTM model...")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        ]

        # Add model checkpoint if logs directory exists
        if self.model_paths['logs']:
            checkpoint_path = Path(self.model_paths['logs']) / 'best_model.keras'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    str(checkpoint_path),
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min'
                )
            )

        try:
            history = self.models['lstm'].fit(
                [seq_features, static_seq],
                targets,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )

            print("\nModel training completed successfully!")
            self.save_models()

        except Exception as e:
            print(f"\nError during training: {str(e)}")
            print("\nDebugging information:")
            print(f"Sequence features shape: {seq_features.shape}")
            print(f"Static features shape: {static_seq.shape}")
            print(f"Targets shape: {targets.shape}")
            raise

    def save_models(self) -> None:
        """Save the trained models and components."""
        # Create model directory if it doesn't exist
        save_dir = Path(self.model_paths['base'])
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save RF model
        rf_path = save_dir / 'rf_model.joblib'
        joblib.dump(self.models['rf'], rf_path)

        # Save LSTM model
        lstm_path = save_dir / 'lstm_model.keras'
        self.models['lstm'].save(lstm_path)

        # Save feature engineering component
        feature_path = save_dir / 'feature_engineering.joblib'
        joblib.dump(self.feature_engineering, feature_path)

        print(f"Models and components saved to {save_dir}")


if __name__ == "__main__":
    trainer = HorseRaceModel()
    trainer.train(use_cache=True)