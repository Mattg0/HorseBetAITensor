from pathlib import Path
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import json
import sys

# Project imports
from env_setup import setup_environment, get_database_path
from core.database import Database
from core.api.race_api import RaceAPI

from utils.race_formatter import RaceFormatter
from utils.race_repository import RaceRepository
from model.Claude.claude_predict_race import HorseRacePredictor
from model.Claude.models.incremental_trainer import IncrementalTrainer
from core.db.daily_race_to_historical import DailyRaceToHistorical


class DeltaTrainer:
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize delta trainer with configuration."""
        self.config_path = config_path
        self.config = setup_environment(config_path)
        self.start_date = None
        self.end_date = None

        # Use active_db from config
        active_db = self.config.get('active_db', '2years')  # Default to 2years if not specified
        print(f"Using active database: {active_db}")

        # Initialize database with active_db
        self.db = Database(config_path, db_name=active_db)
        self.historical_db_path = get_database_path(self.config, active_db)

        # Initialize other components
        self.api = RaceAPI(api_key=self.config['aspiturf']['api_key'])
        self.repository = RaceRepository(self.db)
        self.formatter = RaceFormatter()
        self.predictor = HorseRacePredictor()

    def get_latest_historical_date(self) -> str:
        """Get the latest date from historical database."""
        try:
            # Use the Database class to get connection
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT MAX(jour) 
                    FROM Course
                """)
                latest_date = cursor.fetchone()[0]

            if not latest_date:
                print("No data found in historical database, using default start date")
                # Default to 7 days ago if no historical data
                return (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

            # Add one day to the latest date to avoid overlap
            next_date = (datetime.strptime(latest_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"Latest historical date: {latest_date}, starting from: {next_date}")
            return next_date

        except Exception as e:
            print(f"Error getting latest historical date: {e}")
            # Return default date on error
            return (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    def store_race(self, formatted_data: Dict) -> bool:
        """
        Store race data in Course table and results in Resultats table.

        Args:
            formatted_data: Dictionary containing formatted race and ordre_arrivee data
        """
        try:
            if not formatted_data:
                return False

            race = formatted_data['race']
            ordre_arrivee = formatted_data['ordre_arrivee']
            comp_id = race['comp']

            with self.db.get_connection() as conn:
                # Store in Course table
                conn.execute("""
                    INSERT INTO Course (
                        comp, jour, hippodrome, meteo, dist, corde,
                        natpis, pistegp, typec, temperature, forceVent,
                        directionVent, nebulosite, participants
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    race['comp'],
                    race['jour'],
                    race['hippodrome'],
                    race.get('meteo', ''),
                    race.get('dist', 0),
                    race.get('corde', ''),
                    race.get('natpis', ''),
                    race.get('pistegp', ''),
                    race.get('typec', ''),
                    race.get('temperature', 0),
                    race.get('forceVent', 0),
                    race.get('directionVent', ''),
                    race.get('nebulosite', ''),
                    race['participants']
                ))

                # Store in Resultats table
                if ordre_arrivee:
                    conn.execute("""
                        INSERT INTO Resultats (comp, ordre_arrivee, created_at)
                        VALUES (?, ?, datetime('now'))
                    """, (comp_id, json.dumps(ordre_arrivee)))

                conn.commit()
                return True

        except Exception as e:
            print(f"Error storing race {race.get('comp')}: {e}")
            return False

    def verify_and_fetch_races(self, start_date: str, end_date: str) -> List[Dict]:
        """Verify races in DB and fetch any missing ones."""
        print(f"\nVerifying races from {start_date} to {end_date}...")

        try:
            # Get existing races from DB
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT comp, jour 
                    FROM daily_races 
                    WHERE jour BETWEEN ? AND ?
                """, (start_date, end_date))
                existing_races = {row[0] for row in cursor.fetchall()}

            # Fetch all races for date range from API
            missing_races = []
            processed_comps = set()  # Track processed comp IDs
            current_date = datetime.strptime(start_date, "%Y-%m-%d")
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

            while current_date <= end_date_obj:
                date_str = current_date.strftime("%Y-%m-%d")
                daily_races = self.api.fetch_daily_races(date_str)

                for race in daily_races:
                    try:
                        comp_id = race['numcourse']['comp']
                        # Skip if race exists or has already been processed
                        if comp_id in existing_races or comp_id in processed_comps:
                            continue

                        print(f"Found missing race: {comp_id} on {date_str}")
                        processed_comps.add(comp_id)  # Mark as processed

                        # Fetch full race details
                        race_details = self.api.fetch_race_details(
                            date_str,
                            race['numcourse']['reun'],
                            race['numcourse']['prix']
                        )

                        if race_details:
                            formatted_race = self.formatter.format_race_data(race_details)
                            if formatted_race:
                                missing_races.append(formatted_race)

                    except Exception as e:
                        print(f"Error processing race from API: {e}")
                        continue

                current_date += timedelta(days=1)

            print(f"Found {len(missing_races)} unique missing races")
            return missing_races

        except Exception as e:
            print(f"Error verifying races: {e}")
            return []
    def store_races(self, races: List[Dict]) -> bool:
        """Store races in database."""
        try:
            for race in races:
                self.store_race(race)
                self.repository.store_race(race)
            return True
        except Exception as e:
            print(f"Error storing races: {e}")
            return False
    def verify_predictions(self) -> List[int]:
        """Verify all races have full order predictions."""
        print("\nVerifying predictions...")
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT c.comp 
                    FROM Course c
                    LEFT JOIN predictions p ON c.comp = p.comp 
                        AND p.bet_type = 'full'
                    WHERE p.comp IS NULL AND c.jour BETWEEN ? AND ?
                """,(self.start_date, self.end_date))
                missing_predictions = [row[0] for row in cursor.fetchall()]
                print(f"Found {len(missing_predictions)} races missing predictions")
                return missing_predictions
        except Exception as e:
            print(f"Error verifying predictions: {e}")
            return []
    def predict_full_order(self, comp_id: int) -> Optional[Dict]:
        """Generate full order prediction for a race."""
        try:
            # Get race details
            with self.db.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM Course WHERE comp = ?", (comp_id,))
                race = cursor.fetchone()
                if not race:
                    print(f"Race {comp_id} not found in database")
                    return None
            # Generate prediction
            result = self.predictor.predict_race(
                comp_id,
                bet_type='full',
                return_sequence_only=True,
                model_type='combined'
            )
            if result and isinstance(result, dict):
                return {
                    'comp': comp_id,
                    'bet_type': 'full',
                    'sequence': result['sequence'],
                    'confidence': result['confidence'],
                    'model_type': 'combined'
                }
            return None
        except Exception as e:
            print(f"Error predicting race {comp_id}: {e}")
            return None
    def store_prediction(self, prediction: Dict) -> bool:
        """Store prediction in database."""
        try:
            with self.db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO predictions (
                        comp, bet_type, sequence, confidence, model_type
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    prediction['comp'],
                    prediction['bet_type'],
                    prediction['sequence'],
                    prediction['confidence'],
                    prediction['model_type']
                ))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error storing prediction: {e}")
            return False
    def ensure_complete_data(self, start_date: str, end_date: str) -> bool:
        """Ensure all races and predictions are present and complete."""
        try:
            # 1. Verify and fetch missing races
            missing_races = self.verify_and_fetch_races(start_date, end_date)
            if missing_races:
                print("\nStoring missing races...")
                if not self.store_races(missing_races):
                    return False
            # 2. Verify and generate missing predictions
            races_needing_predictions = self.verify_predictions()
            prediction_count = 0
            for comp_id in races_needing_predictions:
                prediction = self.predict_full_order(comp_id)
                if prediction and self.store_prediction(prediction):
                    prediction_count += 1
                    print(f"Stored prediction for race {comp_id}")
            # 3. Final verification
            final_check = self.verify_predictions()
            if not final_check:
                print("All races and predictions are complete")
                return True
            else:
                print(f"Still missing predictions for races: {final_check}")
                return False
        except Exception as e:
            print(f"Error ensuring complete data: {e}")
            return False

    def run(self) -> bool:
        """Execute the complete delta training process."""
        try:
            print(f"\nStarting delta training process for period: {self.start_date} to {self.end_date}")

            # Step 1: Ensure all data is complete
            if not self.ensure_complete_data(self.start_date, self.end_date):
                print("Failed to ensure complete data")
                return False

            # Step 2: Get races with results for training
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT DISTINCT c.comp
                    FROM Course c
                    JOIN Resultats r ON c.comp = r.comp
                    WHERE c.jour BETWEEN ? AND ?
                      AND EXISTS (SELECT 1 FROM predictions p WHERE p.comp = c.comp)
                """, (self.start_date, self.end_date))
                delta_races = [row[0] for row in cursor.fetchall()]

            if not delta_races:
                print("No new races with predictions and results to process")
                #return True

            print(f"Found {len(delta_races)} complete races to process")

            # Step 3: Retrain model
            trainer = IncrementalTrainer()
            if not trainer.perform_incremental_training(delta_races):
                print("Error retraining model")
                return False

            print("\nDelta training completed successfully!")
            return True

        except Exception as e:
            print(f"Error during delta training: {e}")
            return False
def main():
    """Main entry point for delta training."""
    try:
        trainer = DeltaTrainer()
        # Get start date from latest historical data
        trainer.start_date = '2025-02-14'
        # Use current date as end date
        #end_date = datetime.now().strftime("%Y-%m-%d")
        trainer.end_date = '2025-02-15'
        print(f"\nRunning delta training from {trainer.start_date} to {trainer.end_date}")
        success = trainer.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Fatal error in delta training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()