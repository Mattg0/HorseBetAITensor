from pathlib import Path
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import json

from env_setup import setup_environment, get_database_path
from core.database import Database
from model.Claude.claude_predict_race import HorseRacePredictor


class IncrementalTrainer:
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize incremental trainer."""
        self.config = setup_environment(config_path)
        self.db = Database(config_path)
        self.predictor = HorseRacePredictor()

    def get_training_data(self, comp_ids: List[int]) -> Optional[pd.DataFrame]:
        """Get race data for specific comp_ids with results."""
        try:
            print(f"\nFetching training data for {len(comp_ids)} races...")
            with self.db.get_connection() as conn:
                placeholders = ','.join('?' * len(comp_ids))
                query = f"""
                    SELECT 
                        c.*,
                        r.ordre_arrivee,
                        r.created_at as result_created
                    FROM Course c
                    JOIN Resultats r ON c.comp = r.comp
                    WHERE c.comp IN ({placeholders})
                    ORDER BY c.jour
                """

                df = pd.read_sql_query(query, conn, params=comp_ids)
                if df.empty:
                    print("No data found for specified races")
                    return None

                print(f"Successfully fetched {len(df)} races")
                return df

        except Exception as e:
            print(f"Error fetching training data: {e}")
            return None

    def process_race_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Process race data into training features and targets."""
        try:
            features = []
            targets = []

            for _, race in df.iterrows():
                try:
                    # Parse participants and results
                    participants = json.loads(race['participants'])
                    ordre_arrivee = json.loads(race['ordre_arrivee'])

                    # Sort results by finishing position
                    sorted_results = sorted(ordre_arrivee, key=lambda x: x['narrivee'])
                    actual_sequence = [int(h['numero']) for h in sorted_results]

                    # Extract features for each horse
                    race_features = {
                        'comp': race['comp'],
                        'dist': race['dist'],
                        'corde': race['corde'],
                        'temperature': race['temperature'],
                        'participants': participants,
                        'natpis': race['natpis'],
                        'typec': race['typec']
                    }

                    features.append(race_features)
                    targets.append(actual_sequence)

                except Exception as e:
                    print(f"Error processing race {race['comp']}: {e}")
                    continue

            if not features:
                raise ValueError("No valid features extracted from races")

            return np.array(features), np.array(targets)

        except Exception as e:
            print(f"Error in process_race_data: {e}")
            raise

    def update_models(self, features: np.ndarray, targets: np.ndarray) -> bool:
        """Update both RF and LSTM models with new data."""
        try:
            print("\nUpdating models with new data...")

            # Prepare data for RF model
            rf_features = self.predictor.prepare_rf_features(features)
            if self.predictor.rf_model is not None:
                self.predictor.rf_model.partial_fit(rf_features, targets)

            # Prepare data for LSTM model
            seq_features, static_features = self.predictor.prepare_sequence_data(
                pd.DataFrame(features), is_training=True
            )
            if self.predictor.lstm_model is not None:
                self.predictor.lstm_model.fit(
                    [seq_features, static_features],
                    targets,
                    epochs=5,
                    batch_size=32,
                    validation_split=0.2
                )

            return True

        except Exception as e:
            print(f"Error updating models: {e}")
            return False

    def perform_incremental_training(self, comp_ids: List[int]) -> bool:
        """Perform incremental training on specified races."""
        try:
            if not comp_ids:
                print("No races specified for training")
                return False

            # Get training data
            training_data = self.get_training_data(comp_ids)
            if training_data is None:
                return False

            # Process data into features and targets
            features, targets = self.process_race_data(training_data)

            # Update models
            if not self.update_models(features, targets):
                print("Failed to update models")
                return False

            print(f"Successfully trained on {len(features)} races")
            return True

        except Exception as e:
            print(f"Error in incremental training: {e}")
            return False