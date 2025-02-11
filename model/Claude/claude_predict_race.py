from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from typing import Dict, Tuple, Union
from env_setup import setup_environment, get_model_paths
from core.format_coursedata import main as get_race_data
from core.prep_history_data import main as get_historical_races
from utils.cache_manager import CacheManager

class HorseRacePredictor:
    def __init__(self, config_path: str = 'config.yaml', model_name: str = 'hybrid', sequence_length: int = 5):
        """Initialize the predictor with configuration."""
        self.config = setup_environment(config_path)
        self.model_paths = get_model_paths(self.config, model_name)
        self.sequence_length = sequence_length
        self.rf_model = None
        self.lstm_model = None
        self.feature_engineering = None
        self.historical_data = None

        # Initialize cache manager
        cache_dir = Path(self.config['paths']['cache'])
        self.cache_manager = CacheManager(cache_dir)

        self._load_model_components()
        self._load_historical_data()

    def _load_model_components(self) -> None:
        """Load the trained models and components."""
        base_path = Path(self.model_paths['base'])

        # Load RF model
        rf_path = base_path / 'rf_model.joblib'
        if not rf_path.exists():
            raise FileNotFoundError(f"RF model not found at {rf_path}")
        self.rf_model = joblib.load(rf_path)

        # Load LSTM model
        lstm_path = base_path / 'lstm_model.keras'
        if not lstm_path.exists():
            raise FileNotFoundError(f"LSTM model not found at {lstm_path}")
        self.lstm_model = tf.keras.models.load_model(str(lstm_path))

        # Load feature engineering
        feature_path = base_path / 'feature_engineering.joblib'
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature engineering not found at {feature_path}")
        self.feature_engineering = joblib.load(feature_path)

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

            # Process the data
            self.historical_data = self.feature_engineering.extract_all_features(historical_df)

            # Cache the processed data
            self.cache_manager.save(self.historical_data, 'historical_processed_data')

        except Exception as e:
            print(f"Error loading historical data: {e}")
            raise

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

    def prepare_sequence_data(self, df: pd.DataFrame, is_prediction: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for LSTM prediction."""
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

        # Convert features to numeric type
        for col in sequential_features + static_features_list:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())

        # Group by horse
        for horse_id in df['idche'].unique():
            try:
                # Get historical data for this horse
                horse_history = self.historical_data[self.historical_data['idche'] == horse_id]
                current_race = df[df['idche'] == horse_id]

                if not horse_history.empty:
                    # Get sequential features from history
                    hist_features = horse_history[sequential_features].values.astype(np.float32)

                    # Add current race data (with position=0)
                    current_features = current_race[sequential_features].values.astype(np.float32)
                    seq_features = np.vstack([hist_features, current_features])
                else:
                    # No history available, use only current race data
                    seq_features = current_race[sequential_features].values.astype(np.float32)

                # Get static features from current race
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
                    # Take most recent sequence
                    seq_features = seq_features[-self.sequence_length:]

                sequences.append(seq_features)
                static_features.append(static_feat)

            except (ValueError, TypeError) as e:
                print(f"Error processing horse {horse_id}: {e}")
                continue

        if not sequences:
            raise ValueError("No valid sequences could be created")

        sequences = np.array(sequences, dtype=np.float32)
        static_features = np.array(static_features, dtype=np.float32)

        return sequences, static_features

    def _format_prediction_output(self, results: pd.DataFrame) -> None:
        """Format and print prediction results in a clean way."""
        print("\n" + "=" * 50)
        print("RACE PREDICTION RESULTS")
        print("=" * 50)

        # Format the main prediction table
        prediction_table = results[['horse_name', 'predicted_rank', 'odds']].copy()
        prediction_table['odds'] = prediction_table['odds'].map('{:.1f}'.format)
        print(prediction_table.to_string(index=False))

        # Print the numero sequence
        print("\nPredicted finishing order (numbers):")
        numero_sequence = results['numero'].astype(str).tolist()
        print("-".join(numero_sequence))
        print("-" * 50)

    def _analyze_model_agreement(self, results: pd.DataFrame) -> Dict:
        """Analyze agreement between RF and LSTM predictions."""
        try:
            # Calculate custom correlation
            rank_correlation = self._calculate_correlation(
                results['rf_prediction'],
                results['lstm_prediction']
            )

            # Calculate agreement on top 3
            rf_top3 = set(results.nsmallest(3, 'rf_prediction')['horse_name'])
            lstm_top3 = set(results.nsmallest(3, 'lstm_prediction')['horse_name'])
            top3_agreement = len(rf_top3.intersection(lstm_top3)) / 3

            # Calculate variances
            rf_variance = results['rf_prediction'].std()
            lstm_variance = results['lstm_prediction'].std()

            agreement_data = {
                'rank_correlation': rank_correlation,
                'top3_agreement': top3_agreement,
                'rf_variance': rf_variance,
                'lstm_variance': lstm_variance
            }

            print("\nMODEL AGREEMENT ANALYSIS")
            print("-" * 50)
            print(f"Rank Correlation: {rank_correlation:.2f}")
            print(f"Top 3 Agreement: {top3_agreement:.1%}")
            print(f"Model Variances - RF: {rf_variance:.2f}, LSTM: {lstm_variance:.2f}")

            if rank_correlation > 0.7 and top3_agreement >= 0.67:
                print("Status: HIGH agreement between models")
            elif rank_correlation < 0.3 or top3_agreement == 0:
                print("Status: LOW agreement between models - treat with caution")
            else:
                print("Status: MODERATE agreement between models")

            return agreement_data

        except Exception as e:
            print("Warning: Could not complete model agreement analysis")
            return {
                'rank_correlation': 0.0,
                'top3_agreement': 0.0,
                'rf_variance': 0.0,
                'lstm_variance': 0.0
            }

    def analyze_confidence(self, predictions: pd.DataFrame, agreement_data: Dict) -> None:
        """Analyze and display prediction confidence metrics."""
        try:
            # Calculate basic metrics
            favorite_confidence = 1.0 / predictions.iloc[0]['odds'] if predictions.iloc[0]['odds'] > 0 else 0.0
            top_3_odds_sum = predictions.head(3)['odds'].sum()
            prediction_spread = predictions['predicted_position'].std()

            print("\nPREDICTION CONFIDENCE ANALYSIS")
            print("-" * 50)

            # Print core metrics
            print("Core Metrics:")
            print(f"  Favorite Win Probability: {favorite_confidence:.1%}")
            print(f"  Top 3 Combined Odds: {top_3_odds_sum:.1f}")
            print(f"  Prediction Spread: {prediction_spread:.1f}")
            print(f"  Model Agreement: {agreement_data['rank_correlation']:.2f}")

            # Calculate confidence score
            confidence_score = (
                    min(1.0, favorite_confidence / 0.5) * 20 +
                    min(1.0, 10 / max(0.1, top_3_odds_sum)) * 20 +
                    min(1.0, prediction_spread / 2.0) * 20 +
                    agreement_data['rank_correlation'] * 20 +
                    min(1.0, (agreement_data['rf_variance'] + agreement_data['lstm_variance']) / 2) * 20
            )

            # Determine confidence level
            if confidence_score >= 80:
                confidence_level = "HIGH"
            elif confidence_score >= 60:
                confidence_level = "MEDIUM"
            else:
                confidence_level = "LOW"

            print(f"\nOverall Confidence Level: {confidence_level}")
            print(f"Confidence Score: {confidence_score:.1f}/100")
            print("=" * 50)

        except Exception as e:
            print(f"\nWarning: Error in confidence analysis: {e}")
            print("Unable to calculate confidence metrics")

    def predict_race(self, comp_id: int, bet_type: str = 'tierce', return_sequence_only: bool = False) -> Union[pd.DataFrame, str]:
        """
        Predict race outcomes using both RF and LSTM models.

        Args:
            comp_id: Competition ID
            bet_type: Type of bet ('tierce', 'quarte', or 'quinte')
            return_sequence_only: If True, returns only the sequence string (e.g., "1-2-3")

        Returns:
            Either DataFrame with full predictions or string with number sequence
        """
        # Validate bet type
        bet_type = bet_type.lower()
        if bet_type not in ['tierce', 'quarte', 'quinte']:
            raise ValueError("bet_type must be one of: 'tierce', 'quarte', 'quinte'")

        # Map bet type to number of positions
        bet_positions = {
            'tierce': 3,
            'quarte': 4,
            'quinte': 5
        }
        num_positions = bet_positions[bet_type]

        if not return_sequence_only:
            print(f"\nPredicting outcomes for race {comp_id} ({bet_type.upper()})...")

        # Get race data
        race_data_json = get_race_data(comp_id)
        if race_data_json is None:
            raise ValueError(f"No data found for race {comp_id}")

        try:
            # Parse JSON data
            import json
            race_data = json.loads(race_data_json)

            # Create DataFrame from race data
            participants = race_data['participants']
            df = pd.DataFrame(participants)

            # Add race info to each row
            for key, value in race_data['course_info'].items():
                df[key] = value

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

            # Sort and filter for bet type
            results = results.sort_values('predicted_position')
            results['predicted_rank'] = range(1, len(results) + 1)
            bet_results = results.head(num_positions)

            # Generate sequence
            sequence = "-".join(bet_results['numero'].astype(str).tolist())

            if return_sequence_only:
                return sequence

            # Print full prediction output
            print(f"\n{bet_type.upper()} Prediction:")
            print(bet_results[['horse_name', 'predicted_rank', 'odds']].to_string(index=False))
            print(f"\n{bet_type.upper()} sequence:")
            print(sequence)

            return bet_results

        except Exception as e:
            if not return_sequence_only:
                print(f"Error during prediction: {str(e)}")
            raise




if __name__ == "__main__":
    # Example usage
    predictor = HorseRacePredictor()

    # Get the test competition ID from config
    test_comp_id = predictor.config['test_comp_id']

    try:
        # Make predictions for the test race
        predictions = predictor.predict_race(test_comp_id,bet_type='tierce', return_sequence_only=True)
        print(predictions)
        print("\nPrediction completed successfully!")

    except Exception as e:
        print(f"Error during prediction: {str(e)}")

