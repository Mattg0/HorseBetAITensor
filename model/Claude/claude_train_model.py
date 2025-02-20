from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from typing import Tuple
from core.prep_history_data import main as get_historical_races
from env_setup import setup_environment, get_model_paths, get_cache_path
from features import FeatureEngineering
from models.architectures import create_hybrid_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.ensemble import RandomForestRegressor
from utils.cache_manager import CacheManager


class HorseRaceModel:
    def __init__(self, config_path: str = 'config.yaml', model_name: str = 'hybrid',
                 sequence_length: int = 5):
        """Initialize the model with configuration."""
        self.config = setup_environment(config_path)
        self.model_paths = get_model_paths(self.config, model_name)
        self.sequence_length = sequence_length
        self.models = None
        self.rf_model = None  # Add this
        self.lstm_model = None  # Add this
        self.feature_engineering = FeatureEngineering()
        self.history = None

        # Get active database type
        self.db_type = self.config['active_db']

        # Initialize cache manager with correct paths
        cache_dir = Path(get_cache_path(self.config, 'training_data', self.db_type)).parent
        self.cache_manager = CacheManager(cache_dir)
        print(f"Initialized with database type: {self.db_type}")
        print(f"Using cache directory: {cache_dir}")

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
        print(f"\nLoading historical race data for {self.db_type} database...")

        # Get cache paths for both historical and training data
        historical_cache = get_cache_path(self.config, 'historical_data', self.db_type)
        training_cache = get_cache_path(self.config, 'training_data', self.db_type)

        # Try to load processed data from cache
        if use_cache:
            print(f"Attempting to load cached data from: {training_cache}")
            cached_data = self.cache_manager.load(training_cache)
            if cached_data is not None:
                print("Found cached training data")
                df_features = cached_data
            else:
                print("No cached data found, processing historical data...")
                # Try to load historical data from cache first
                historical_data = self.cache_manager.load(historical_cache)
                if historical_data is None:
                    print("Loading raw historical data...")
                    historical_data = get_historical_races()
                    self.cache_manager.save(historical_data, historical_cache)

                print("Extracting features...")
                df_features = self.feature_engineering.extract_all_features(historical_data)
                self.cache_manager.save(df_features, training_cache)
        else:
            print("Cache disabled, processing historical data...")
            df_historical = get_historical_races()
            df_features = self.feature_engineering.extract_all_features(df_historical)

        # Prepare static features for RF
        print("\nPreparing features for training...")
        static_columns = self.feature_engineering.get_feature_columns()
        static_features = df_features[static_columns].astype(float)

        print("\nPreparing sequence data...")
        seq_features, static_seq, targets = self.prepare_sequence_data(df_features)

        if len(seq_features) == 0:
            raise ValueError("No valid sequences found in the data")

        # Create hybrid model
        print("\nCreating model...")
        self.models = create_hybrid_model(
            sequence_length=self.sequence_length,
            seq_feature_dim=seq_features.shape[2],
            static_feature_dim=static_seq.shape[1]
        )

        print(f"\nTraining models using {self.db_type} database...")

        # Train RF model
        print("\nTraining Random Forest model...")
        print(f"Models dictionary contents: {self.models}")  # Debug print
        if 'rf' not in self.models:
            raise ValueError("RF model not found in created models")

        self.rf_model = self.models['rf']
        print(f"RF model type before training: {type(self.rf_model)}")

        if self.rf_model is None:
            raise ValueError("RF model is None after assignment")

        self.rf_model.fit(static_features, df_features['position'])
        print(f"RF model type after training: {type(self.rf_model)}")

        # Train LSTM model
        print("\nTraining LSTM model...")
        self.lstm_model = self.models['lstm']
        if self.lstm_model is None:
            raise ValueError("LSTM model is None after assignment")

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        ]

        # Add model checkpoint if logs directory exists
        if self.model_paths['logs']:
            checkpoint_path = Path(self.model_paths['logs']) / f'best_model_{self.db_type}.keras'
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
                epochs=1,
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
        save_dir = Path(self.model_paths['model_path'])
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving models to: {save_dir}")

        # Save RF model with verification
        rf_path = save_dir / self.model_paths['artifacts']['rf_model']
        print(f"Saving RF model of type: {type(self.rf_model)}")
        if self.rf_model is None:
            raise ValueError("Cannot save RF model - model is None")
        joblib.dump(self.rf_model, rf_path)

        # Verify the save
        loaded_check = joblib.load(rf_path)
        if loaded_check is None:
            raise ValueError("RF model save verification failed - loaded model is None")
        print(f"Saved and verified RF model to: {rf_path}")

        # Save LSTM model
        lstm_path = save_dir / self.model_paths['artifacts']['lstm_model']
        self.models['lstm'].save(lstm_path)
        print(f"Saved LSTM model to: {lstm_path}")

        # Save feature engineering state
        feature_path = save_dir / self.model_paths['artifacts']['feature_engineer']
        feature_eng_state = {
            'feature_columns': self.feature_engineering.get_feature_columns(),
            'position_history': self.feature_engineering.position_history,
            'jockey_stats': self.feature_engineering.jockey_stats,
            'n_jobs': self.feature_engineering.n_jobs
        }
        joblib.dump(feature_eng_state, feature_path)
        print(f"Saved feature engineering state to: {feature_path}")

        print(f"\nAll models and components saved successfully to {save_dir}")


if __name__ == "__main__":
    trainer = HorseRaceModel()
    trainer.train(use_cache=True)