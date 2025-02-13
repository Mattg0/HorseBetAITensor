from pathlib import Path
from datetime import datetime
from core.api.race_api import RaceAPI
from utils.race_formatter import RaceFormatter
from utils.race_repository import RaceRepository
from core.database import Database
from scripts.TelegramBotPrediction import TelegramNotifier
from typing import List, Dict, Optional


class RacePredictionOrchestrator:
    def __init__(self, config_path: str = 'config.yaml'):
        from env_setup import setup_environment
        self.config = setup_environment(config_path)

        # Initialize components
        db_config = next((db for db in self.config['databases'] if db['name'] == 'full'), None)
        if not db_config:
            raise ValueError("Full database configuration not found")

        self.db = Database(config_path)
        self.api = RaceAPI(api_key="8cdfGeF4pHeSOPv05dPnVyGaghL2")
        self.repository = RaceRepository(self.db)
        self.formatter = RaceFormatter()
        self.telegram = TelegramNotifier(
            self.config['telegram']['bot_token'],
            self.config['telegram']['chat_id']
        )

    def fetch_and_store_races(self):
        """Fetch today's races from API and store in the database."""
        today = datetime.now().strftime("%Y-%m-%d")

        # Fetch today's races
        daily_races = self.api.fetch_daily_races(today)

        # Store each race
        for race_data in daily_races:
            try:
                # Fetch full race details
                race_details = self.api.fetch_race_details(
                    today,
                    race_data['numcourse']['reun'],
                    race_data['numcourse']['prix']
                )

                if race_details:
                    # Format race data
                    formatted_race = self.formatter.format_race_data(race_details)
                    if formatted_race:
                        # Store race data
                        self.repository.store_race(formatted_race)

            except Exception as e:
                print(f"Error storing race {race_data['numcourse']['comp']}: {e}")
                continue

    def process_stored_races(self):
        """Process all stored unpredicted races."""
        # Fetch unpredicted races from database
        date='2025-02-13'
        unpredicted_races = self.repository.get_unpredicted_races(date)

        # Process each race
        for race_data in unpredicted_races:
            try:
                # Make predictions
                from model.Claude.claude_predict_race import HorseRacePredictor
                predictor = HorseRacePredictor()

                predictions = {}
                for bet_type in ['tierce', 'quarte', 'quinte']:
                    result = predictor.predict_race(
                        race_data['comp'],
                        bet_type=bet_type,
                        return_sequence_only=True
                    )
                    predictions[bet_type] = result
                    self.repository.store_prediction(
                        race_data['comp'],
                        bet_type,
                        result['sequence'],
                        result['confidence']
                    )

                # Mark race as predicted
                self.repository.mark_race_predicted(race_data['comp'])

                # Send Telegram notification
                message = self._format_prediction_message(race_data, predictions)
                self.telegram.send_message(message)

            except Exception as e:
                print(f"Error processing race {race_data['comp']}: {e}")
                continue

    def _format_prediction_message(self, race: Dict, predictions: Dict) -> str:
        """Format predictions into a Telegram message."""
        message = (
            f"🏇 <b>{race['hippodrome']} - {race['heure']}</b>\n"
            f"📍 {race['prixnom']}\n\n"
        )

        for bet_type, prediction in predictions.items():
            emoji = "🥇" if bet_type == 'tierce' else "🏆" if bet_type == 'quarte' else "👑"
            confidence = prediction['confidence']
            confidence_emoji = "🎯" if confidence >= 75 else "⚠️" if confidence <= 25 else "📊"

            message += (
                f"{emoji} {bet_type.upper()}: {prediction['sequence']}\n"
                f"{confidence_emoji} Confidence: {confidence}%\n\n"
            )

        return message


if __name__ == "__main__":
    orchestrator = RacePredictionOrchestrator()
    orchestrator.fetch_and_store_races()
    orchestrator.process_stored_races()