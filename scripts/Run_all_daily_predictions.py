from pathlib import Path
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Union, Tuple
import json

from env_setup import setup_environment, get_database_path
from core.database import Database
from model.Claude.claude_predict_race import HorseRacePredictor


class DailyPredictionHandler:
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the daily prediction handler."""
        self.config = setup_environment(config_path)
        self.db = Database(config_path)
        self.predictor = HorseRacePredictor()

    def get_races_to_predict(self, comp_ids: Optional[List[int]] = None) -> List[Dict]:
        """
        Get list of races that need predictions.
        If comp_ids is provided, only get those races; otherwise get all unpredicted races.
        """
        try:
            conn = self.db.get_connection()

            # Base query for race information
            base_query = """
                SELECT DISTINCT dr.comp, dr.hippodrome, dr.jour, dr.reun, dr.prix, dr.heure, 
                       dr.prixnom, dr.quinte
                FROM daily_races dr
                LEFT JOIN predictions p ON dr.comp = p.comp
                WHERE p.comp IS NULL
            """

            # Add comp_ids filter if provided
            if comp_ids:
                placeholders = ','.join('?' * len(comp_ids))
                query = f"{base_query} AND dr.comp IN ({placeholders})"
                cursor = conn.execute(query, comp_ids)
            else:
                cursor = conn.execute(base_query)

            # Fetch and format results
            races = []
            for row in cursor.fetchall():
                races.append({
                    'comp': row[0],
                    'hippodrome': row[1],
                    'jour': row[2],
                    'reun': row[3],
                    'prix': row[4],
                    'heure': row[5],
                    'prixnom': row[6],
                    'quinte': row[7]
                })

            return races

        except Exception as e:
            print(f"Error getting races to predict: {e}")
            return []
        finally:
            conn.close()

    def predict_race(self, race: Dict) -> Optional[List[Dict]]:
        """
        Generate predictions for a single race.
        Returns list of predictions with sequence and confidence for each bet type.
        """
        try:
            comp_id = race['comp']
            is_quinte = race['quinte']

            # Determine bet types based on race type
            bet_types = ['tierce', 'quarte', 'quinte'] if is_quinte else ['quarte']

            predictions = []
            for bet_type in bet_types:
                try:
                    result = self.predictor.predict_race(
                        comp_id,
                        bet_type=bet_type,
                        return_sequence_only=True,
                        model_type='combined'  # Using combined model type by default
                    )

                    if result and isinstance(result, dict):
                        predictions.append({
                            'bet_type': 'full',  # Always use 'full' as per requirement
                            'sequence': result['sequence'],
                            'confidence': result['confidence'],
                            'model_type': 'combined'  # Using combined model as default
                        })
                except Exception as e:
                    print(f"Error predicting {bet_type} for race {comp_id}: {e}")
                    continue

            return predictions if predictions else None

        except Exception as e:
            print(f"Error in race prediction: {e}")
            return None

    def store_predictions(self, comp_id: int, predictions: List[Dict]) -> bool:
        """
        Store prediction results in the database according to schema:
        comp INTEGER NOT NULL,
        bet_type TEXT NOT NULL,
        sequence TEXT NOT NULL,
        confidence FLOAT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        model_type TEXT
        """
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()

            for pred in predictions:
                cursor.execute("""
                    INSERT INTO predictions (
                        comp,
                        bet_type,
                        sequence,
                        confidence,
                        model_type
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    comp_id,
                    pred['bet_type'],
                    pred['sequence'],
                    pred['confidence'],
                    pred['model_type']
                ))

            conn.commit()
            return True

        except Exception as e:
            print(f"Error storing predictions for race {comp_id}: {e}")
            return False
        finally:
            conn.close()

    def process_races(self, comp_ids: Optional[List[int]] = None) -> Tuple[int, int]:
        """
        Process predictions for specified races or all unpredicted races.
        Returns tuple of (success_count, total_count).
        """
        # Get races that need predictions
        races = self.get_races_to_predict(comp_ids)
        if not races:
            print("No races found to predict")
            return (0, 0)

        print(f"\nProcessing predictions for {len(races)} races...")
        success_count = 0

        for race in races:
            comp_id = race['comp']
            try:
                print(f"\nPredicting race {comp_id} at {race['hippodrome']}")

                # Generate predictions
                predictions = self.predict_race(race)
                if not predictions:
                    print(f"No predictions generated for race {comp_id}")
                    continue

                # Store predictions
                if self.store_predictions(comp_id, predictions):
                    success_count += 1
                    print(f"Successfully processed race {comp_id}")
                    # Print stored predictions for verification
                    for pred in predictions:
                        print(f"Stored {pred['bet_type']} prediction: {pred['sequence']} "
                              f"(confidence: {pred['confidence']}%, model: {pred['model_type']})")
                else:
                    print(f"Failed to store predictions for race {comp_id}")

            except Exception as e:
                print(f"Error processing race {comp_id}: {e}")
                continue

        return (success_count, len(races))


def run_predictions(comp_ids: Optional[List[int]] = None) -> bool:
    """
    Main function to run predictions for specified races or all unpredicted races.
    """
    try:
        handler = DailyPredictionHandler()
        success_count, total_count = handler.process_races(comp_ids)

        print(f"\nPrediction process completed:")
        print(f"Successfully processed {success_count} out of {total_count} races")

        return success_count == total_count

    except Exception as e:
        print(f"Error in prediction process: {e}")
        return False


if __name__ == "__main__":
    import sys

    # Handle command line arguments for comp IDs
    comp_ids = None
    if len(sys.argv) > 1:
        try:
            comp_ids = [int(comp_id) for comp_id in sys.argv[1:]]
            print(f"Processing specific races: {comp_ids}")
        except ValueError:
            print("Error: Competition IDs must be numbers")
            sys.exit(1)

    success = run_predictions(comp_ids)
    sys.exit(0 if success else 1)