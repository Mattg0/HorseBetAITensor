from pathlib import Path
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json
from typing import Dict, Tuple, Union
from env_setup import setup_environment, get_model_paths,get_cache_path
from core.format_dailyrace_data import main as get_race_data
from core.prep_history_data import main as get_historical_races
from model.Claude.features import FeatureEngineering
from utils.cache_manager import CacheManager
from scripts.FeatureHandler import FeatureHandler
import scipy


class HorseRacePredictor:
    def __init__(self, config_path: str = 'config.yaml', sequence_length: int = 5):
        """Initialize the predictor with configuration."""
        self.config = setup_environment(config_path)
        self.sequence_length = sequence_length
        self.rf_model = None
        self.lstm_model = None
        self.feature_engineering = None
        self.historical_data = None

        # Get model paths
        self.model_paths = get_model_paths(self.config, 'claude')
        self.model_dir = Path(self.model_paths['model_path'])

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found at {self.model_dir}")

        # Initialize cache manager
        cache_dir = Path(self.config['rootdir']) / self.config['paths']['cache']
        self.cache_manager = CacheManager(cache_dir)

        self._load_model_components()
        self._load_historical_data()

    def _load_model_components(self) -> None:
        """Load the trained models and components."""
        try:
            rf_path = Path(self.model_paths['model_path']) / self.model_paths['artifacts']['rf_model']
            if rf_path.exists():
                loaded_rf = joblib.load(rf_path)
                if loaded_rf is None:
                    raise ValueError("joblib.load returned None for RF model")
                self.rf_model = loaded_rf
            else:
                raise FileNotFoundError(f"RF model not found at {rf_path}")

            # Verify RF model more thoroughly
            if not hasattr(self.rf_model, 'predict'):
                print(f"DEBUG: RF model lacks predict method. Available attributes: {dir(self.rf_model)}")
                raise ValueError("Loaded RF model lacks predict method")


            # Load LSTM model
            lstm_path = Path(self.model_paths['model_path']) / self.model_paths['artifacts']['lstm_model']
            if not lstm_path.exists():
                raise FileNotFoundError(f"LSTM model not found at {lstm_path}")
            self.lstm_model = tf.keras.models.load_model(str(lstm_path))

            # Load feature engineering state
            feature_path = Path(self.model_paths['model_path']) / self.model_paths['artifacts']['feature_engineer']
            print(f"Looking for feature engineering at: {feature_path}")
            if not feature_path.exists():
                raise FileNotFoundError(f"Feature engineering not found at {feature_path}")

            # Create new instance and restore state
            self.feature_engineering = FeatureEngineering()
            feature_state = joblib.load(feature_path)

            # Restore state
            for key, value in feature_state.items():
                setattr(self.feature_engineering, key, value)
            print("Successfully loaded feature engineering")

            # Verify models are loaded correctly
            if self.rf_model is None:
                raise ValueError("RF model failed to load properly")
            if self.lstm_model is None:
                raise ValueError("LSTM model failed to load properly")
            if self.feature_engineering is None:
                raise ValueError("Feature engineering failed to load properly")
        except Exception as e:
            print(f"\nERROR during model loading: {str(e)}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Model path directory exists: {Path(self.model_paths['model_path']).exists()}")
            print(f"Model directory contents: {list(Path(self.model_paths['model_path']).glob('*'))}")
            # Try to read the file contents
            try:
                with open(rf_path, 'rb') as f:
                    first_bytes = f.read(100)
                    print(f"First few bytes of RF model file: {first_bytes}")
            except Exception as read_error:
                print(f"Could not read RF model file: {read_error}")
            raise


    def _load_historical_data(self) -> None:
        """Load and process historical race data with caching."""
        # Try to load from cache first
        cached_data = self.cache_manager.load('historical_processed_data')
        if cached_data is not None:
            self.historical_data = cached_data
            return

        print("Loading and processing historical data...")
        try:
            # Load raw historical data
            historical_df = get_historical_races()

            # Hash categorical columns before feature extraction
            historical_df = self._hash_categorical_columns(historical_df)

            # Process the data
            self.historical_data = self.feature_engineering.extract_all_features(historical_df)

            # Cache the processed data
            self.cache_manager.save(self.historical_data, 'historical_processed_data')

        except Exception as e:
            print(f"Error loading historical data: {e}")
            raise

    def calculate_confidence_score(self, results: pd.DataFrame, num_positions: int) -> float:
        """Calculate confidence score based on various factors."""
        try:
            # Get top N horses based on bet type
            top_horses = results.head(num_positions)

            # 1. Model Agreement Score (0-25 points)
            rf_preds = top_horses['rf_prediction'].values
            lstm_preds = top_horses['lstm_prediction'].values

            # Use rankings instead of raw predictions
            rf_ranks = scipy.stats.rankdata(rf_preds)
            lstm_ranks = scipy.stats.rankdata(lstm_preds)
            model_agreement_score = max(0, min(25, (1 - np.mean(np.abs(rf_ranks - lstm_ranks)) / num_positions) * 25))

            # 2. Odds Confidence (0-25 points)
            odds = top_horses['odds'].values
            if not np.any(np.isnan(odds)):
                odds_ranks = scipy.stats.rankdata(odds)
                pred_ranks = scipy.stats.rankdata(range(len(odds)))
                odds_score = max(0, min(25, (1 - np.mean(np.abs(odds_ranks - pred_ranks)) / num_positions) * 25))
            else:
                odds_score = 12.5  # Neutral score if odds are missing

            # 3. Position Gap Score (0-25 points)
            position_gaps = np.diff(top_horses['predicted_position'].values)
            if len(position_gaps) > 0 and not np.any(np.isnan(position_gaps)):
                avg_gap = np.mean(position_gaps)
                gap_score = max(0, min(25, (avg_gap / 0.5) * 25))
            else:
                gap_score = 12.5  # Neutral score if gaps can't be calculated

            # 4. Prediction Stability (0-25 points)
            if not np.any(np.isnan(rf_preds)) and not np.any(np.isnan(lstm_preds)):
                # Calculate variance of normalized rankings
                variances = [np.var([r, l]) / num_positions for r, l in zip(rf_ranks, lstm_ranks)]
                stability_score = max(0, min(25, (1 - np.mean(variances)) * 25))
            else:
                stability_score = 12.5  # Neutral score if predictions are missing

            # Combine scores
            total_confidence = (
                    model_agreement_score +
                    odds_score +
                    gap_score +
                    stability_score
            )

            return round(total_confidence)

        except Exception as e:
            print(f"Warning: Error calculating confidence score: {e}")
            return 50
    def _get_latest_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get latest statistics for horses and jockeys from historical data."""
        df_copy = df.copy()

        # Get the latest statistics for each horse
        for horse_id in df_copy['idche'].unique():
            horse_history = self.historical_data[self.historical_data['idche'] == horse_id]
            if not horse_history.empty:
                latest_stats = horse_history.sort_values('jour').iloc[-1]

                # Copy rolling statistics for horse
                rolling_cols = [
                    'avg_pos_3', 'avg_pos_5', 'avg_pos_10',
                    'win_rate_3', 'win_rate_5', 'win_rate_10',
                    'place_rate_3', 'place_rate_5', 'place_rate_10'
                ]
                for col in rolling_cols:
                    if col in latest_stats:
                        df_copy.loc[df_copy['idche'] == horse_id, col] = latest_stats[col]
                    else:
                        df_copy[col] = 0.0  # Default value if not available

        # Get the latest statistics for each jockey
        for jockey_id in df_copy['idJockey'].unique():
            jockey_history = self.historical_data[self.historical_data['idJockey'] == jockey_id]
            if not jockey_history.empty:
                latest_stats = jockey_history.sort_values('jour').iloc[-1]

                # Copy jockey statistics
                jockey_cols = ['jockey_win_rate', 'jockey_place_rate', 'jockey_avg_pos']
                for col in jockey_cols:
                    if col in latest_stats:
                        df_copy.loc[df_copy['idJockey'] == jockey_id, col] = latest_stats[col]
                    else:
                        df_copy[col] = 0.5  # Default value if not available

        return df_copy

    def _hash_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hash categorical columns consistently with training data."""
        import hashlib

        categorical_cols = ['natpis', 'typec', 'meteo', 'corde']
        df = df.copy()

        # Hash categorical columns
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(
                    lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % (10 ** 8)
                )

        # Convert date to timestamp
        if 'jour' in df.columns:
            df['jour'] = pd.to_datetime(df['jour']).astype(int) // 10 ** 9

        return df

    def _safe_convert_to_float(self, value: Union[str, float]) -> float:
        """Safely convert a value to float, handling non-numeric values."""
        try:
            float_val = float(value)
            if np.isfinite(float_val):
                return float_val
            return 0.0
        except (ValueError, TypeError):
            return 0.0

    def prepare_sequence_data(self, df: pd.DataFrame, is_prediction: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for LSTM prediction with proper normalization."""
        sequences = []
        static_features = []

        # Define features
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

        # Handle missing position field for prediction
        if is_prediction and 'position' not in df.columns:
            df['position'] = 0  # Placeholder for current race

        # Normalize numeric features
        for col in sequential_features + static_features_list:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if col not in ['musique_fault_prone', 'normalized_odds']:  # Skip binary features
                    df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
                df[col] = df[col].fillna(0)

        # Group by horse
        for horse_id in df['idche'].unique():
            try:
                # Get historical data for this horse
                horse_history = self.historical_data[self.historical_data['idche'] == horse_id]
                current_race = df[df['idche'] == horse_id]

                if not horse_history.empty:
                    # Get sequential features from history
                    hist_features = horse_history[sequential_features].values.astype(np.float32)
                    # Add current race data
                    current_features = current_race[sequential_features].values.astype(np.float32)
                    seq_features = np.vstack([hist_features, current_features])
                else:
                    seq_features = current_race[sequential_features].values.astype(np.float32)

                # Get static features
                static_feat = current_race[static_features_list].iloc[0].values.astype(np.float32)

                # Pad sequence if needed
                if len(seq_features) < self.sequence_length:
                    pad_length = self.sequence_length - len(seq_features)
                    seq_features = np.pad(
                        seq_features,
                        ((pad_length, 0), (0, 0)),
                        mode='constant',
                        constant_values=0
                    )
                else:
                    seq_features = seq_features[-self.sequence_length:]

                sequences.append(seq_features)
                static_features.append(static_feat)

            except Exception as e:
                print(f"Error processing horse {horse_id}: {e}")
                continue

        if not sequences:
            raise ValueError("No valid sequences could be created")

        return np.array(sequences, dtype=np.float32), np.array(static_features, dtype=np.float32)

    def _calculate_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate rank correlation between two arrays."""
        try:
            return np.corrcoef(x, y)[0, 1]
        except:
            return 0.0

    def predict_race(self, comp_id: int, bet_type: str = 'tierce', return_sequence_only: bool = False,
                     model_type: str = 'combined') -> Union[pd.DataFrame, Dict, str]:
        """
        Predict race outcomes using selected model type.

        Args:
            comp_id: Competition ID
            bet_type: Type of bet ('tierce', 'quarte', 'quinte', or 'full')
            return_sequence_only: If True, returns only the sequence string
            model_type: Type of model to use ('combined', 'rf', or 'lstm')
        """
        try:
            # Initialize feature handler
            feature_handler = FeatureHandler()

            # Get race data
            race_data_json = get_race_data(comp_id)
            if race_data_json is None:
                raise ValueError(f"No data found for race {comp_id}")

            # Parse JSON data
            race_data = json.loads(race_data_json)
            participants = race_data['participants']
            df = pd.DataFrame(participants)

            # Add race info to each row
            for key, value in race_data['course_info'].items():
                df[key] = value

            # Hash categorical columns before feature extraction
            df = self._hash_categorical_columns(df)

            # Extract base features
            df_features = self.feature_engineering.extract_all_features(df, is_training=False)

            # Prepare all features with the feature handler
            df_features = feature_handler.prepare_features(
                df_features,
                history_df=self.historical_data if hasattr(self, 'historical_data') else None
            )

            # Verify all required features are present
            required_features = self.feature_engineering.get_feature_columns()
            if not feature_handler.verify_features(df_features, required_features):
                raise ValueError("Missing or invalid features after preparation")

            # Get features for prediction
            static_features = df_features[required_features].astype(float)

            # Prepare sequence data
            seq_features, static_seq = self.prepare_sequence_data(df_features, is_prediction=True)

            # Make predictions
            rf_predictions = self.rf_model.predict(static_features)
            lstm_raw_predictions = self.lstm_model.predict([seq_features, static_seq], verbose=0).flatten()

            # Post-process predictions
            lstm_predictions = np.clip(lstm_raw_predictions, 1, len(df))
            lstm_predictions = scipy.stats.rankdata(lstm_predictions)

            # Combine predictions based on model_type
            if model_type == 'rf':
                final_predictions = rf_predictions
            elif model_type == 'lstm':
                final_predictions = lstm_predictions
            else:  # combined
                final_predictions = 0.6 * scipy.stats.rankdata(rf_predictions) + 0.4 * lstm_predictions

            # Create results DataFrame
            results = pd.DataFrame({
                'horse_id': df['idche'],
                'horse_name': df['cheval'],
                'numero': df['numero'],
                'predicted_position': final_predictions,
                'rf_prediction': rf_predictions,
                'lstm_prediction': lstm_predictions,
                'odds': df['cotedirect']
            })

            # Sort and prepare output
            results = results.sort_values('predicted_position')
            results['predicted_rank'] = range(1, len(results) + 1)

            # Determine number of positions to return
            if bet_type == 'full':
                num_positions = len(results)
            else:
                bet_positions = {'tierce': 3, 'quarte': 4, 'quinte': 5}
                num_positions = bet_positions[bet_type]

            # Get results for requested positions
            bet_results = results.head(num_positions)

            # Generate sequence and calculate confidence
            sequence = "-".join(bet_results['numero'].astype(str).tolist())
            confidence_score = self.calculate_confidence_score(results, num_positions)

            if return_sequence_only:
                return {
                    'sequence': sequence,
                    'confidence': confidence_score,
                    'model_type': model_type,
                    'num_horses': len(bet_results)
                }

            return bet_results, confidence_score

        except Exception as e:
            print(f"Error predicting race {comp_id}: {str(e)}")
            raise


    def _process_race_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process race data for prediction."""
        try:
            # Hash categorical columns
            df = self._hash_categorical_columns(df)

            # Extract and process features
            df_features = self.feature_engineering.extract_all_features(df, is_training=False)
            df_features = self._get_latest_stats(df_features)

            # Get static features for RF
            static_features = df_features[self.feature_engineering.get_feature_columns()].astype(float)

            # Get sequential features for LSTM
            seq_features, static_seq = self.prepare_sequence_data(df_features, is_prediction=True)

            # Make predictions
            rf_predictions = self.rf_model.predict(static_features)
            lstm_predictions = self.lstm_model.predict([seq_features, static_seq], verbose=0)

            # Combine predictions
            combined_predictions = 0.6 * rf_predictions + 0.4 * lstm_predictions.flatten()

            # Create results DataFrame
            results = pd.DataFrame({
                'horse_id': df['idche'],
                'horse_name': df['cheval'],
                'numero': df['numero'],
                'predicted_position': combined_predictions,
                'rf_prediction': rf_predictions,
                'lstm_prediction': lstm_predictions.flatten(),
                'odds': df['cotedirect']
            })

            # Sort by predicted position
            results = results.sort_values('predicted_position')
            results['predicted_rank'] = range(1, len(results) + 1)

            return results

        except Exception as e:
            print(f"Error processing race data: {str(e)}")
            raise

    def _get_top_predictions(self, df: pd.DataFrame, num_positions: int) -> pd.DataFrame:
        """Get top N predictions from processed data."""
        return df.head(num_positions)

if __name__ == "__main__":
    # Example usage
    predictor = HorseRacePredictor()

    # Get the test competition ID from config
    test_comp_id = '1412739'

    try:
        # Make predictions for the test race
        predictions = predictor.predict_race(test_comp_id, bet_type='full', return_sequence_only=False)
        print(predictions)
        print("\nPrediction completed successfully!")

    except Exception as e:
        print(f"Error during prediction: {str(e)}")