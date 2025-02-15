from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import List, Dict
import sys
import sqlite3

from env_setup import setup_environment
from model.Claude.claude_predict_race import HorseRacePredictor


class DailyPredictionRunner:
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize prediction runner."""
        self.config = setup_environment(config_path)
        self.predictor = HorseRacePredictor()
        self.db = self._get_db_connection()

    def _get_db_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        db_config = next((db for db in self.config['databases'] if db['name'] == 'full'), None)
        if not db_config:
            raise ValueError("Full database configuration not found")
        db_path = Path(self.config['rootdir']) / db_config['path']
        return sqlite3.connect(db_path)

    def get_todays_races(self) -> List[Dict]:
        """Get all races for today from the database."""
        today = datetime.now().strftime("%Y-%m-%d")
        cursor = self.db.execute("""
            SELECT comp, quinte
            FROM daily_races
            WHERE jour = ?
            ORDER BY heure
        """, (today,))
        return [{'comp': row[0], 'quinte': row[1]} for row in cursor.fetchall()]

    def store_prediction(self, comp: int, bet_type: str, sequence: str, confidence: float):
        """Store prediction results in database."""
        cursor = self.db.execute("""
            INSERT INTO predictions (comp, bet_type, sequence, confidence, timestamp)
            VALUES (?, ?, ?, ?, datetime('now'))
        """, (comp, bet_type, sequence, confidence))
        self.db.commit()

    def run_daily_predictions(self):
        """Run predictions for all of today's races."""
        print("Starting daily predictions...")
        races = self.get_todays_races()

        for race in races:
            comp_id = race['comp']
            is_quinte = bool(race['quinte'])

            try:
                # Determine bet types for this race
                bet_types = ['tierce', 'quarte', 'quinte'] if is_quinte else ['tierce']

                for bet_type in bet_types:
                    print(f"\nPredicting {bet_type.upper()} for race {comp_id}")

                    # Get prediction
                    result = self.predictor.predict_race(
                        comp_id,
                        bet_type=bet_type,
                        return_sequence_only=True
                    )

                    if result:
                        sequence = result['sequence']
                        confidence = result['confidence']

                        # Store prediction
                        self.store_prediction(comp_id, bet_type, sequence, confidence)
                        print(f"Stored {bet_type} prediction: {sequence} (confidence: {confidence}%)")

            except Exception as e:
                print(f"Error predicting race {comp_id}: {str(e)}")
                continue

        print("\nDaily predictions completed!")


def main():
    runner = DailyPredictionRunner()
    runner.run_daily_predictions()


if __name__ == "__main__":
    main()