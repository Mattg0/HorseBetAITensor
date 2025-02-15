import os
import sys
import pandas as pd
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import numpy as np
from scipy import stats

# First get the config and set up paths
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

# Now we can import our project modules
class QuintePlusEvaluator:
    @staticmethod
    def evaluate_quinte_prediction(prediction: str, actual: str) -> Dict[str, bool]:
        """
        Evaluate a Quinté+ prediction against actual results.
        Returns detailed win types according to official Quinté+ rules.
        """
        if not prediction or not actual:
            return {
                'exact_order': False,
                'any_order': False,
                'bonus_4': False,
                'bonus_3': False
            }

        pred_numbers = prediction.split('-')
        actual_numbers = actual.split('-')

        # Ensure we have 5 numbers for both
        pred_numbers = pred_numbers[:5]
        actual_numbers = actual_numbers[:5]

        # Pad if necessary
        while len(pred_numbers) < 5:
            pred_numbers.append('0')
        while len(actual_numbers) < 5:
            actual_numbers.append('0')

        # Check exact order (all 5 horses in correct order)
        exact_order = pred_numbers == actual_numbers

        # Check if all 5 horses are present in any order
        any_order = set(pred_numbers) == set(actual_numbers)

        # Check Bonus 4 (first 4 horses in any order)
        pred_set_4 = set(pred_numbers[:4])
        actual_set_4 = set(actual_numbers[:4])
        bonus_4 = len(pred_set_4.intersection(actual_set_4)) == 4

        # Check Bonus 3 (first 3 horses in any order)
        pred_set_3 = set(pred_numbers[:3])
        actual_set_3 = set(actual_numbers[:3])
        bonus_3 = len(pred_set_3.intersection(actual_set_3)) == 3

        return {
            'exact_order': exact_order,
            'any_order': any_order,
            'bonus_4': bonus_4,
            'bonus_3': bonus_3
        }


class EnhancedPerformanceTester:
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the enhanced performance tester."""
        self.config = setup_environment(config_path)
        self.db_path = self._get_db_path()

        # Ensure we're in the right directory
        os.chdir(self.config['rootdir'])

    def _get_db_path(self) -> Path:
        """Get database path from config."""
        db_config = next((db for db in self.config['databases'] if db['name'] == 'full'), None)
        if not db_config:
            raise ValueError("Full database configuration not found")
        return Path(self.config['rootdir']) / db_config['path']

    def _connect_to_db(self) -> sqlite3.Connection:
        """Create an optimized database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.execute('PRAGMA journal_mode = WAL')
        conn.execute('PRAGMA cache_size = -2000')
        conn.execute('PRAGMA synchronous = NORMAL')
        return conn

    def fetch_daily_races(self, date: str) -> pd.DataFrame:
        """Fetch races from daily_races table for a specific date."""
        query = """
        SELECT dr.*, r.ordre_arrivee, r.comp as result_comp
        FROM daily_races dr
        LEFT JOIN Resultats r ON dr.comp = r.comp
        WHERE dr.jour = ? AND r.ordre_arrivee IS NOT NULL
        """

        with self._connect_to_db() as conn:
            df = pd.read_sql_query(query, conn, params=(date,))

            # Process ordre_arrivee to create actual_order column
            def format_ordre_arrivee(ordre):
                try:
                    if pd.isna(ordre) or not ordre:
                        return ''
                    # Parse JSON array and extract arrival order
                    arrival_data = json.loads(ordre)
                    # Sort by narrivee and extract numero
                    sorted_numbers = sorted(arrival_data, key=lambda x: int(x.get('narrivee', 99)))
                    actual_order = '-'.join(str(x.get('numero', '')) for x in sorted_numbers)
                    return actual_order
                except Exception as e:
                    print(f"Error processing ordre_arrivee: {e}")
                    return ''

            # Add the actual_order column by processing ordre_arrivee
            df['actual_order'] = df['ordre_arrivee'].apply(format_ordre_arrivee)

            # Drop any rows where we couldn't process the actual order
            df = df[df['actual_order'] != '']

            return df

    def _ensure_predictions_table(self):
        """Ensure predictions table exists in database."""
        with self._connect_to_db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    comp INTEGER NOT NULL,
                    bet_type TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    sequence TEXT NOT NULL,
                    confidence REAL,
                    created_at DATETIME NOT NULL,
                    FOREIGN KEY (comp) REFERENCES daily_races(comp)
                )
            """)

    def store_prediction(self, comp: int, bet_type: str, model_type: str,
                         sequence: str, confidence: float, timestamp: datetime) -> None:
        """Store prediction in the predictions table."""
        self._ensure_predictions_table()
        with self._connect_to_db() as conn:
            conn.execute("""
                INSERT INTO predictions (
                    comp, bet_type, model_type, sequence, confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (comp, bet_type, model_type, sequence, confidence, timestamp))

    def evaluate_prediction(self, prediction: str, actual_results: str,
                            bet_type: str) -> Dict[str, float]:
        """Evaluate a prediction against actual results."""
        try:
            pred_numbers = prediction.split('-')
            actual_numbers = actual_results.strip().split('-') if actual_results else []

            positions = {'tierce': 3, 'quarte': 4, 'quinte': 5}
            num_positions = positions.get(bet_type, 3)

            pred_numbers = pred_numbers[:num_positions]
            actual_numbers = actual_numbers[:num_positions]

            exact_matches = sum(p == a for p, a in zip(pred_numbers, actual_numbers))
            numbers_found = sum(p in actual_numbers for p in pred_numbers)

            return {
                'exact_matches': exact_matches,
                'numbers_found': numbers_found,
                'accuracy': exact_matches / num_positions,
                'hit_rate': numbers_found / num_positions
            }
        except Exception as e:
            print(f"Error evaluating prediction: {e}")
            return {
                'exact_matches': 0,
                'numbers_found': 0,
                'accuracy': 0.0,
                'hit_rate': 0.0
            }

    def analyze_predictions(self, date: str, predictor, model_type: str = 'combined') -> Dict:
        """Analyze predictions for a specific date using specified model type."""
        daily_races = self.fetch_daily_races(date)
        if daily_races.empty:
            return {"error": f"No races found for date {date}"}

        results = []
        total_predictions = 0
        successful_predictions = 0

        for _, race in daily_races.iterrows():
            comp_id = race['comp']
            is_quinte = bool(race['quinte'])

            bet_types = ['tierce', 'quarte', 'quinte'] if is_quinte else ['tierce']

            for bet_type in bet_types:
                try:
                    # Get prediction with specified model type
                    prediction_result = predictor.predict_race(
                        comp_id,
                        bet_type=bet_type,
                        return_sequence_only=True,
                        model_type=model_type
                    )

                    if not prediction_result:
                        continue

                    sequence = prediction_result['sequence']
                    confidence = prediction_result['confidence']
                    prediction_time = datetime.now()

                    # Store prediction
                    self.store_prediction(
                        comp_id,
                        bet_type,
                        model_type,
                        sequence,
                        confidence,
                        prediction_time
                    )

                    evaluation = self.evaluate_prediction(
                        sequence,
                        race['actual_order'],
                        bet_type
                    )

                    results.append({
                        'comp_id': comp_id,
                        'bet_type': bet_type,
                        'model_type': model_type,
                        'prediction': sequence,
                        'actual': race['actual_order'],
                        'confidence': confidence,
                        **evaluation
                    })

                    total_predictions += 1
                    if evaluation['numbers_found'] > 0:
                        successful_predictions += 1

                except Exception as e:
                    print(f"Error processing race {comp_id} for {bet_type}: {e}")
                    continue

        if not results:
            return {"error": "No predictions were made"}

        df_results = pd.DataFrame(results)

        metrics = {
            'total_predictions': total_predictions,
            'successful_predictions': successful_predictions,
            'success_rate': (successful_predictions / total_predictions) * 100 if total_predictions > 0 else 0,
            'average_accuracy': df_results['accuracy'].mean() * 100,
            'average_hit_rate': df_results['hit_rate'].mean() * 100,
            'average_confidence': df_results['confidence'].mean(),
            'high_confidence_success_rate': df_results[
                                                df_results['confidence'] >= 75
                                                ]['hit_rate'].mean() * 100 if len(
                df_results[df_results['confidence'] >= 75]) > 0 else 0
        }

        return {
            'metrics': metrics,
            'detailed_results': results
        }

    def generate_report(self, analysis_results: Dict, model_type: str) -> str:
        """Generate a detailed performance report."""
        if 'error' in analysis_results:
            return f"Error: {analysis_results['error']}"

        metrics = analysis_results['metrics']
        detailed_results = analysis_results['detailed_results']

        report = [
            f"Performance Analysis Report ({model_type.upper()} Model)",
            "=" * 50,
            f"\nOverall Metrics:",
            f"Total Predictions: {metrics['total_predictions']}",
            f"Successful Predictions: {metrics['successful_predictions']}",
            f"Success Rate: {metrics['success_rate']:.2f}%",
            f"Average Accuracy: {metrics['average_accuracy']:.2f}%",
            f"Average Hit Rate: {metrics['average_hit_rate']:.2f}%",
            f"Average Confidence: {metrics['average_confidence']:.2f}%",
            f"High Confidence Success Rate: {metrics['high_confidence_success_rate']:.2f}%"
        ]

        if detailed_results:
            report.extend([
                "\nDetailed Results:",
                "-" * 50
            ])

            for result in detailed_results[:10]:  # Show first 10 results
                report.extend([
                    f"\nRace {result['comp_id']} ({result['bet_type'].upper()}):",
                    f"Prediction: {result['prediction']}",
                    f"Actual: {result['actual']}",
                    f"Confidence: {result['confidence']:.2f}%",
                    f"Exact Matches: {result['exact_matches']}",
                    f"Numbers Found: {result['numbers_found']}"
                ])

        return "\n".join(report)

def main(model_name: str = 'claude', date: str = None, model_type: str = 'combined'):
    """Main function to run enhanced performance testing."""
    try:
        # Initialize predictor
        from model.Claude.claude_predict_race import HorseRacePredictor
        predictor = HorseRacePredictor()

        # Initialize tester
        tester = EnhancedPerformanceTester()

        # Use provided date or default to today
        test_date = date or datetime.now().strftime("%Y-%m-%d")

        print(f"Running performance analysis for date: {test_date} using {model_type} model")

        # Run analysis with specified model type
        results = tester.analyze_predictions(test_date, predictor, model_type)

        # Generate and print report - pass model_type to generate_report
        if 'error' not in results:
            report = tester.generate_report(results, model_type)
            print("\n" + report)
            return results
        else:
            print(f"\nError: {results['error']}")
            return None

    except Exception as e:
        print(f"Error running performance test: {str(e)}")
        raise





if __name__ == "__main__":
    import sys

    model_type = sys.argv[2] if len(sys.argv) > 2 else 'combined'
    date = sys.argv[1] if len(sys.argv) > 1 else None
    main(date='2025-02-13')