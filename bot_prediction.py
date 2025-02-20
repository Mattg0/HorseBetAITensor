from pathlib import Path
from datetime import datetime
import sqlite3
import json
import time
import threading
from typing import List, Dict, Optional

# Import local modules
from core.api.race_api import RaceAPI
from utils.race_formatter import RaceFormatter
from utils.race_repository import RaceRepository
from core.database import Database
from utils.TelegramNotifier import TelegramNotifier
from env_setup import setup_environment
from model.Claude.claude_predict_race import HorseRacePredictor


class RacePredictionOrchestrator:
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the race prediction orchestrator."""
        # Load configuration
        self.config = setup_environment(config_path)
        self.db = Database(config_path)
        self.api = RaceAPI(api_key="8cdfGeF4pHeSOPv05dPnVyGaghL2")
        self.repository = RaceRepository(self.db)
        self.formatter = RaceFormatter()

        # Initialize predictor
        self.predictor = HorseRacePredictor()

        # Initialize telegram notifier
        self.telegram = TelegramNotifier(
            self.config['telegram']['bot_token'],
            self.config['telegram']['chat_id']
        )
        # Set up callback handler - IMPORTANT: This assigns the handler
        print("Setting up callback handler")  # Debug print
        self.telegram.callback_handler = self.handle_prediction_request

    def handle_prediction_request(self, comp_id: int) -> None:
        """Handle prediction request for a specific race."""
        print(f"\nHandling prediction request for comp_id: {comp_id}")  # Debug print
        try:
            # Get race details from database
            with self.db.get_connection() as conn:
                cursor = conn.execute('''
                    SELECT hippodrome, heure, prixnom, quinte
                    FROM daily_races
                    WHERE comp = ?
                ''', (comp_id,))
                race_info = cursor.fetchone()

            if not race_info:
                print(f"Race {comp_id} not found in database")  # Debug print
                self.telegram.send_message(f"Race {comp_id} not found in database.")
                return

            hippo, heure, prixnom, is_quinte = race_info
            print(f"Found race: {hippo} - {heure} - {prixnom} (Quinté: {is_quinte})")  # Debug print

            # Determine bet types based on race type
            bet_types = ['tierce', 'quarte', 'quinte'] if is_quinte else ['quarte']
            print(f"Will predict for bet types: {bet_types}")  # Debug print

            # Get predictions for each bet type
            predictions = {}
            for bet_type in bet_types:
                try:
                    print(f"Getting prediction for {bet_type}")  # Debug print
                    result = self.predictor.predict_race(
                        comp_id,
                        bet_type=bet_type,
                        return_sequence_only=True
                    )
                    print(f"Prediction result for {bet_type}: {result}")  # Debug print
                    if result and isinstance(result, dict):
                        predictions[bet_type] = result
                except Exception as e:
                    print(f"Error getting prediction for {bet_type}: {e}")
                    continue

            if predictions:
                # Format message
                message = self._format_prediction_message(
                    hippo, heure, prixnom, is_quinte, predictions
                )
                print(f"Sending prediction message:\n{message}")  # Debug print
                self.telegram.send_message(message)
            else:
                print("No predictions available")  # Debug print
                self.telegram.send_message(
                    f"No predictions available for race {comp_id}"
                )

        except Exception as e:
            error_message = f"Error processing prediction for race {comp_id}: {str(e)}"
            print(f"Error: {error_message}")  # Debug print
            self.telegram.send_message(error_message)

            if predictions:
                # Format message
                message = self._format_prediction_message(
                    hippo, heure, prixnom, is_quinte, predictions
                )
                self.telegram.send_message(message)
            else:
                self.telegram.send_message(
                    f"No predictions available for race {comp_id}"
                )

        except Exception as e:
            error_message = f"Error processing prediction for race {comp_id}: {str(e)}"
            print(error_message)
            self.telegram.send_message(error_message)

    def _format_prediction_message(
            self, hippo: str, heure: str, prixnom: str,
            is_quinte: bool, predictions: Dict
    ) -> str:
        """Format prediction results into a message."""
        message = [
            f"{'🌟 ' if is_quinte else ''}🎯 <b>Pronostics {hippo} - {heure}</b>",
            f"📍 {prixnom}\n"
        ]

        for bet_type, prediction in predictions.items():
            emoji = "🥇" if bet_type == 'tierce' else "🏆" if bet_type == 'quarte' else "👑"
            confidence = prediction.get('confidence', 50)
            confidence_emoji = "🎯" if confidence >= 75 else "⚠️" if confidence <= 25 else "📊"

            message.extend([
                f"{emoji} {bet_type.upper()}: {prediction['sequence']}",
                f"{confidence_emoji} Confidence: {confidence}%\n"
            ])

        return "\n".join(message)

    def fetch_and_store_races(self) -> List[Dict]:
        """
        Fetch today's races from API and process from database.
        1. Get races list from API, filtering non-EUR races immediately
        2. Store new races in DB if they don't exist
        3. Return race data from DB for display
        """
        today = datetime.now().strftime("%Y-%m-%d")
        # First fetch today's races basic info from API
        daily_races = self.api.fetch_daily_races(today)
        if not daily_races:
            print("No races found for today")
            return []

        # Filter out non-EUR races early
        eur_races = []
        for race in daily_races:
            numcourse = race.get('numcourse', {})
            if numcourse.get('devise', 'EUR') != 'EUR':
                print(f"Skipping non-EUR race {numcourse.get('comp')}, currency: {numcourse.get('devise')}")
                continue
            eur_races.append(race)

        if not eur_races:
            print("No EUR races found for today")
            return []

        print(f"Found {len(eur_races)} EUR races out of {len(daily_races)} total races")

        # Check existing races in database
        with self.db.get_connection() as conn:
            cursor = conn.execute('''
                SELECT comp FROM daily_races 
                WHERE jour = ?
            ''', (today,))
            existing_races = {row[0] for row in cursor.fetchall()}

        # Store any new races in database
        for race_data in eur_races:
            try:
                comp_id = race_data['numcourse']['comp']

                # Skip if race already exists
                if comp_id in existing_races:
                    continue

                # Fetch and store new race details
                race_details = self.api.fetch_race_details(
                    today,
                    race_data['numcourse']['reun'],
                    race_data['numcourse']['prix']
                )

                if race_details:
                    formatted_race = self.formatter.format_race_data(race_details)
                    if formatted_race:
                        self.repository.store_race(formatted_race)
                        existing_races.add(comp_id)  # Add to existing races set

            except Exception as e:
                print(f"Error storing race {race_data['numcourse'].get('comp')}: {e}")
                continue

        # Fetch all today's EUR races from database
        with self.db.get_connection() as conn:
            cursor = conn.execute('''
                SELECT dr.*, json_extract(dr.participants, '$') as participants_json
                FROM daily_races dr
                WHERE jour = ?
                ORDER BY reun, prix
            ''', (today,))

            columns = [description[0] for description in cursor.description]
            db_races = []

            for row in cursor.fetchall():
                race_dict = dict(zip(columns, row))

                # Format into the structure expected by TelegramNotifier
                race_info = {
                    'numcourse': {
                        'comp': race_dict['comp'],
                        'hippo': race_dict['hippodrome'],
                        'reun': race_dict['reun'],
                        'prix': race_dict['prix'],
                        'heure': race_dict['heure'],
                        'prixnom': race_dict['prixnom'],
                        'quinte': race_dict['quinte']
                    }
                }
                db_races.append(race_info)

        return db_races
    def send_daily_races_list(self, races: List[Dict]) -> bool:
        """Send the interactive list of today's races to Telegram."""
        return self.telegram.send_daily_races(races)

    def run(self):
        """Main execution flow."""
        try:
            # Fetch and store today's races
            print("Fetching today's races...")
            daily_races = self.fetch_and_store_races()

            if not daily_races:
                print("No races found for today.")
                return

            # Send interactive race list
            print("Sending race list to Telegram...")
            self.send_daily_races_list(daily_races)

            # Start polling for user interactions
            print("Starting interaction polling...")
            self.telegram.start_polling()

            # Keep the script running
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
        except Exception as e:
            print(f"Error in main execution: {e}")
            raise


def main():
    """Entry point for the script."""
    try:
        orchestrator = RacePredictionOrchestrator()
        orchestrator.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()