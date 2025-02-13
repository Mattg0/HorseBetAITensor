import requests
import json
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import sys
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# First get the config
from env_setup import setup_environment, get_model_paths

# Load configuration
config = setup_environment('config.yaml')

# Get paths from config
project_root = Path(config['rootdir'])
model_paths = get_model_paths(config, 'claude')  # Get claude model paths

# Add necessary paths to Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(model_paths['base']) not in sys.path:
    sys.path.insert(0, str(model_paths['base']))

from model.Claude.claude_predict_race import HorseRacePredictor


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        """Initialize Telegram notifier with bot token and chat ID."""
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

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
    def send_message(self, message: str) -> bool:
        """Send a message through Telegram bot."""
        url = f"{self.base_url}/sendMessage"
        data = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML"
        }

        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            print(f"Error sending Telegram message: {e}")
            return False


class BatchPredictor:
    def __init__(self, api_key: str, telegram_bot_token: str, telegram_chat_id: str, config_path: str = 'config.yaml'):
        """Initialize the batch predictor with API key and configuration."""
        self.api_key = api_key
        self.predictor = HorseRacePredictor(config_path=config_path)
        self.telegram = TelegramNotifier(telegram_bot_token, telegram_chat_id)

        # Set up logging
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging configuration."""
        log_dir = Path("logs/batch_predictions")
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def fetch_races(self, date: str) -> Optional[List[Dict]]:
        """Fetch races from the API for a specific date."""
        url = "https://api.aspiturf.com/api"
        params = {
            "uid": self.api_key,
            "jour[]": date
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            races = response.json()
            self.logger.info(f"Successfully fetched {len(races)} races for {date}")
            return races
        except requests.RequestException as e:
            self.logger.error(f"Error fetching races from API: {e}")
            return None

    def predict_single_race(self, race: Dict, bet_types: List[str]) -> Dict:
        """Make predictions for a single race with multiple bet types."""
        comp_id = race['numcourse']['comp']
        hippo = race['numcourse']['hippo']
        course_time = race['numcourse']['heure']
        prix_nom = race['numcourse']['prixnom']

        results = {
            'comp_id': comp_id,
            'hippo': hippo,
            'time': course_time,
            'prix': prix_nom,
            'predictions': {}
        }

        try:
            for bet_type in bet_types:
                prediction_result = self.predictor.predict_race(
                    comp_id,
                    bet_type=bet_type,
                    return_sequence_only=True
                )
                results['predictions'][bet_type] = {
                    'sequence': prediction_result['sequence'],
                    'confidence': prediction_result['confidence']
                }

            self.logger.info(f"Successfully predicted race {comp_id}")
            return results
        except Exception as e:
            self.logger.error(f"Error predicting race {comp_id}: {e}")
            results['error'] = str(e)
            return results

    def format_telegram_message(self, predictions: List[Dict], date: str) -> str:
        """Format predictions into a Telegram message."""
        message = f"🏇 <b>Race Predictions for {date}</b>\n\n"

        for pred in predictions:
            if 'error' in pred:
                message += (
                    f"❌ Race {pred['comp_id']}: Error - {pred['error']}\n"
                )
                continue

            message += (
                f"🎯 <b>{pred['hippo']} - {pred['time']}</b>\n"
                f"📍 {pred['prix']}\n"
            )

            for bet_type, prediction in pred['predictions'].items():
                emoji = "🥇" if bet_type == 'tierce' else "🏆" if bet_type == 'quarte' else "👑"
                confidence = prediction['confidence']
                confidence_emoji = "🎯" if confidence >= 75 else "⚠️" if confidence <= 25 else "📊"
                message += (
                    f"{emoji} {bet_type.upper()}: {prediction['sequence']}\n"
                    f"{confidence_emoji} Confidence: {confidence}%\n"
                )

            message += "\n"

        return message

    def save_predictions(self, predictions: List[Dict], date: str) -> Path:
        """Save predictions to a CSV file."""
        output_dir = Path("predictions")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert predictions to DataFrame
        rows = []
        for pred in predictions:
            comp_id = pred['comp_id']
            if 'error' in pred:
                row = {
                    'comp_id': comp_id,
                    'hippo': pred.get('hippo', ''),
                    'time': pred.get('time', ''),
                    'prix': pred.get('prix', ''),
                    'status': 'error',
                    'error': pred['error']
                }
            else:
                row = {
                    'comp_id': comp_id,
                    'hippo': pred['hippo'],
                    'time': pred['time'],
                    'prix': pred['prix'],
                    'status': 'success',
                    **pred['predictions']
                }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Save to CSV
        output_file = output_dir / f"predictions_{date}.csv"
        df.to_csv(output_file, index=False)
        self.logger.info(f"Saved predictions to {output_file}")

        return output_file

    def process_races(self, races: List[Dict], bet_types: List[str]) -> List[Dict]:
        """Process races in batches with better resource management."""
        predictions = []
        batch_size = 10  # Process 10 races at a time
        total_races = len(races)

        self.logger.info(f"Processing {total_races} races in batches of {batch_size}")

        # Process races in batches
        for i in range(0, total_races, batch_size):
            batch = races[i:i + batch_size]
            batch_predictions = []

            self.logger.info(f"Processing batch {i // batch_size + 1}/{(total_races + batch_size - 1) // batch_size}")

            # Process each batch with limited parallelization
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                for race in batch:
                    try:
                        if 'numcourse' in race and 'comp' in race['numcourse']:
                            futures.append(executor.submit(self.predict_single_race, race, bet_types))
                    except Exception as e:
                        self.logger.error(f"Error submitting race to executor: {e}")

                # Collect results from this batch
                for future in futures:
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout per race
                        if result:
                            batch_predictions.append(result)
                    except Exception as e:
                        self.logger.error(f"Error processing race in batch: {e}")

            # Add batch results to overall predictions
            predictions.extend(batch_predictions)

            # Log progress
            self.logger.info(f"Completed {min(i + batch_size, total_races)}/{total_races} races")

            # Optional: Add a small delay between batches to allow system recovery
            import time
            time.sleep(1)

        return predictions

    def run_batch_predictions(self, date: Optional[str] = None,
                              bet_types: List[str] = ['tierce', 'quarte', 'quinte']) -> Optional[Path]:
        """Run predictions for all races on a specific date."""
        # Use current date if none provided
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        self.logger.info(f"Starting batch predictions for {date}")

        # Fetch races
        races = self.fetch_races(date)
        if not races:
            self.logger.error("No races found for the specified date")
            return None

        # Process races and generate predictions
        predictions = self.process_races(races, bet_types)

        # Save predictions
        output_file = self.save_predictions(predictions, date)

        # Send Telegram message
        message = self.format_telegram_message(predictions, date)
        if self.telegram.send_message(message):
            self.logger.info("Sent predictions to Telegram")
        else:
            self.logger.error("Failed to send predictions to Telegram")

        self.logger.info("Batch predictions completed")
        return output_file


def main():
    """Main function to run batch predictions with enhanced error handling."""
    # Configuration
    config = setup_environment()
    API_KEY = "8cdfGeF4pHeSOPv05dPnVyGaghL2"
    TELEGRAM_BOT_TOKEN = config['telegram']['bot_token']
    TELEGRAM_CHAT_ID = config['telegram']['chat_id']

    # Set up logging for main process
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # Initialize batch predictor
        predictor = BatchPredictor(
            api_key=API_KEY,
            telegram_bot_token=TELEGRAM_BOT_TOKEN,
            telegram_chat_id=TELEGRAM_CHAT_ID
        )

        # Get today's date
        today = datetime.now().strftime("%Y-%m-%d")

        # Run predictions for all races with memory monitoring
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        logger.info(f"Initial memory usage: {initial_memory:.2f} MB")

        output_file = predictor.run_batch_predictions(
            date=today,
            bet_types=['tierce', 'quarte', 'quinte']
        )


        if output_file:
            logger.info(f"Predictions saved to: {output_file}")
        else:
            logger.warning("No predictions generated")

    except Exception as e:
        logger.error(f"Critical error in main process: {str(e)}", exc_info=True)
        # Try to send error notification via Telegram
        try:
            predictor.telegram.send_message(
                f"❌ Error in prediction process:\n{str(e)}\n\nCheck logs for details."
            )
        except:
            logger.error("Failed to send error notification via Telegram")
        raise  # Re-raise the exception for proper error handling


if __name__ == "__main__":
    main()