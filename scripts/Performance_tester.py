import os
import sys
from pathlib import Path
import pandas as pd
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add project root to path
from env_setup import setup_environment, get_database_path
from model.Claude.claude_predict_race import HorseRacePredictor

class PerformanceTester:
    """Analyzes prediction performance for horse races."""

    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the performance tester."""
        self.config = setup_environment(config_path)
        self.db_path = get_database_path(self.config)
        self.predictor = HorseRacePredictor(config_path)

    def _ensure_predictions_table(self) -> None:
        """Ensure predictions table exists in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    comp INTEGER NOT NULL,
                    bet_type TEXT NOT NULL,
                    sequence TEXT NOT NULL,
                    confidence REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(comp, bet_type)
                )
            """)

    def get_races_for_date(self, date: str) -> List[Dict]:
        """Get all completed races for a specific date.

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            List of race dictionaries with results
        """
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT 
                    dr.comp,
                    dr.quinte,
                    dr.hippodrome,
                    dr.heure,
                    dr.prixnom,
                    r.ordre_arrivee
                FROM daily_races dr
                JOIN Resultats r ON dr.comp = r.comp
                WHERE dr.jour = ?
                ORDER BY dr.heure
            """
            cursor = conn.execute(query, (date,))
            columns = [col[0] for col in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_stored_prediction(self, comp_id: int, bet_type: str) -> Optional[Dict]:
        """Get stored prediction for a race if it exists.

        Args:
            comp_id: Race competition ID
            bet_type: Type of bet (tierce, quarte, quinte)

        Returns:
            Dictionary with prediction data or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT sequence, confidence
                FROM predictions
                WHERE comp = ? AND bet_type = ?
            """, (comp_id, bet_type))
            result = cursor.fetchone()

            if result:
                return {
                    'sequence': result[0],
                    'confidence': result[1]
                }
            return None

    def store_prediction(self, comp_id: int, bet_type: str,
                         sequence: str, confidence: float) -> None:
        """Store a prediction in the database.

        Args:
            comp_id: Race competition ID
            bet_type: Type of bet
            sequence: Predicted sequence
            confidence: Prediction confidence score
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO predictions 
                (comp, bet_type, sequence, confidence)
                VALUES (?, ?, ?, ?)
            """, (comp_id, bet_type, sequence, confidence))
            conn.commit()

    def evaluate_prediction(self, prediction: str, actual: str,
                            bet_type: str) -> Dict[str, bool]:
        """Evaluate a prediction against actual results.

        Args:
            prediction: Predicted sequence
            actual: Actual sequence
            bet_type: Type of bet

        Returns:
            Dictionary with evaluation metrics
        """
        pred_nums = prediction.split('-')
        actual_nums = actual.split('-')

        # Handle different bet types
        if bet_type == 'quinte':
            exact_order = pred_nums == actual_nums
            in_top_5 = set(pred_nums[:5]) == set(actual_nums[:5])
            top_4_match = len(set(pred_nums[:4]) & set(actual_nums[:4])) == 4
            top_3_match = len(set(pred_nums[:3]) & set(actual_nums[:3])) == 3

            return {
                'exact_order': exact_order,
                'all_in_top_5': in_top_5,
                'top_4_match': top_4_match,
                'top_3_match': top_3_match
            }
        else:  # tierce/quarte
            positions = 4 if bet_type == 'quarte' else 3
            exact_order = pred_nums[:positions] == actual_nums[:positions]
            correct_horses = set(pred_nums[:positions]) == set(actual_nums[:positions])

            return {
                'exact_order': exact_order,
                'correct_horses': correct_horses
            }

    def analyze_predictions(self, date: str) -> Dict:
        """Analyze predictions for all races on a given date."""
        print(f"\nAnalyzing predictions for {date}")
        races = self.get_races_for_date(date)
        if not races:
            print(f"No races found for {date}")
            return {'error': 'No races found'}

        print(f"Found {len(races)} races to analyze")
        results = []

        for race in races:
            try:
                comp_id = race['comp']
                bet_type = 'quinte' if race['quinte'] else 'tierce'

                prediction = self.predictor.predict_race(
                    comp_id,
                    bet_type=bet_type,
                    return_sequence_only=True
                )

                if prediction:
                    actual_sequence = self._parse_ordre_arrivee(race['ordre_arrivee'])
                    evaluation = self.evaluate_prediction(
                        prediction['sequence'],
                        actual_sequence,
                        bet_type
                    )

                    results.append({
                        'comp_id': comp_id,
                        'hippodrome': race['hippodrome'],
                        'heure': race['heure'],
                        'prixnom': race['prixnom'],
                        'bet_type': bet_type,
                        'prediction': prediction['sequence'],
                        'actual': actual_sequence,
                        'confidence': prediction.get('confidence', 50),
                        **evaluation
                    })
                    print(f"Processed race {comp_id}")

            except Exception as e:
                print(f"Error processing race {comp_id}: {str(e)}")
                continue

        return {
            'metrics': self.calculate_metrics(results),
            'detailed_results': results
        }
    def _parse_ordre_arrivee(self, ordre_arrivee: str) -> str:
        """Parse ordre_arrivee JSON into sequence string."""
        try:
            arrival_data = json.loads(ordre_arrivee)
            sorted_numbers = sorted(arrival_data, key=lambda x: int(x['narrivee']))
            return '-'.join(str(x['numero']) for x in sorted_numbers)
        except Exception as e:
            print(f"Error parsing ordre_arrivee: {e}")
            return ''

    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate success metrics focusing on Quinté+ performance."""
        quinte_results = [r for r in results if r['bet_type'] == 'quinte']

        metrics = {
            'total_races': len(results),
            'quinte_metrics': {
                'total_races': len(quinte_results),
                'exact_order': 0,  # All 5 horses in exact order
                'disorder_5': 0,  # All 5 horses in any order
                'bonus_4': 0,  # First 4 horses in any order
                'bonus_4_plus': 0,  # 4 horses among first 5
                'bonus_3': 0,  # First 3 horses in any order
                'total_hits': 0,  # Any kind of winning combination
                'by_confidence': {
                    'high_conf_success': 0,  # Success rate for high confidence predictions
                    'med_conf_success': 0,  # Success rate for medium confidence predictions
                    'low_conf_success': 0  # Success rate for low confidence predictions
                }
            }
        }

        # Count successes for each prediction
        for result in quinte_results:
            pred_sequence = result['prediction'].split('-')
            actual_sequence = result['actual'].split('-')
            confidence = result['confidence']

            # Check different winning conditions
            exact_order = pred_sequence == actual_sequence
            disorder_5 = set(pred_sequence[:5]) == set(actual_sequence[:5])
            bonus_4 = len(set(pred_sequence[:4]) & set(actual_sequence[:4])) == 4
            bonus_4_plus = len(set(pred_sequence[:4]) & set(actual_sequence[:5])) == 4
            bonus_3 = len(set(pred_sequence[:3]) & set(actual_sequence[:3])) == 3

            # Update counts
            if exact_order:
                metrics['quinte_metrics']['exact_order'] += 1
            if disorder_5:
                metrics['quinte_metrics']['disorder_5'] += 1
            if bonus_4:
                metrics['quinte_metrics']['bonus_4'] += 1
            if bonus_4_plus:
                metrics['quinte_metrics']['bonus_4_plus'] += 1
            if bonus_3:
                metrics['quinte_metrics']['bonus_3'] += 1

            # Count any type of success
            if any([exact_order, disorder_5, bonus_4, bonus_4_plus, bonus_3]):
                metrics['quinte_metrics']['total_hits'] += 1

            # Track success by confidence level
            if confidence >= 75:
                metrics['quinte_metrics']['by_confidence']['high_conf_success'] += 1
            elif confidence >= 50:
                metrics['quinte_metrics']['by_confidence']['med_conf_success'] += 1
            else:
                metrics['quinte_metrics']['by_confidence']['low_conf_success'] += 1

        # Calculate percentages if we have races
        if len(quinte_results) > 0:
            total = len(quinte_results)
            quinte_metrics = metrics['quinte_metrics']

            # Add percentage metrics
            quinte_metrics.update({
                'exact_order_rate': (quinte_metrics['exact_order'] / total) * 100,
                'disorder_5_rate': (quinte_metrics['disorder_5'] / total) * 100,
                'bonus_4_rate': (quinte_metrics['bonus_4'] / total) * 100,
                'bonus_4_plus_rate': (quinte_metrics['bonus_4_plus'] / total) * 100,
                'bonus_3_rate': (quinte_metrics['bonus_3'] / total) * 100,
                'total_success_rate': (quinte_metrics['total_hits'] / total) * 100
            })

        return metrics

    def generate_report(self, metrics: Dict) -> str:
        """Generate a detailed report of the analysis results."""
        report = [
            "\nQuinté+ Performance Analysis",
            "=" * 50,
            f"\nTotal Races Analyzed: {metrics['total_races']}",
            f"Total Quinté+ Races: {metrics['quinte_metrics']['total_races']}",
            "\nSuccess Rates:",
            f"  Exact Order: {metrics['quinte_metrics']['exact_order_rate']:.2f}%",
            f"  All 5 in Disorder: {metrics['quinte_metrics']['disorder_5_rate']:.2f}%",
            f"  Bonus 4 (First 4): {metrics['quinte_metrics']['bonus_4_rate']:.2f}%",
            f"  Bonus 4+ (4 in 5): {metrics['quinte_metrics']['bonus_4_plus_rate']:.2f}%",
            f"  Bonus 3: {metrics['quinte_metrics']['bonus_3_rate']:.2f}%",
            f"  Any Win Type: {metrics['quinte_metrics']['total_success_rate']:.2f}%",
            "\nRaw Counts:",
            f"  Exact Order Wins: {metrics['quinte_metrics']['exact_order']}",
            f"  Disorder 5 Wins: {metrics['quinte_metrics']['disorder_5']}",
            f"  Bonus 4 Wins: {metrics['quinte_metrics']['bonus_4']}",
            f"  Bonus 4+ Wins: {metrics['quinte_metrics']['bonus_4_plus']}",
            f"  Bonus 3 Wins: {metrics['quinte_metrics']['bonus_3']}",
            f"  Total Hits: {metrics['quinte_metrics']['total_hits']}"
        ]

        return "\n".join(report)


def main(date: Optional[str] = None):
    """Run performance analysis for a specific date."""
    try:
        tester = PerformanceTester()

        # Use provided date or default to today
        test_date = date or datetime.now().strftime("%Y-%m-%d")

        # Run analysis
        results = tester.analyze_predictions(test_date)

        # Display results
        if 'error' not in results:
            print("\nAnalysis Results:")
            print("-" * 50)

            metrics = results['metrics']
            print(f"\nTotal Races Analyzed: {metrics['total_races']}")

            # Quinté+ results
            print("\nQuinté+ Performance:")
            q_metrics = metrics['quinte_plus']
            print(f"Total Races: {q_metrics['total']}")
            print(f"Exact Order: {q_metrics.get('exact_order_rate', 0):.2f}%")
            print(f"All in Top 5: {q_metrics.get('all_in_top_5_rate', 0):.2f}%")
            print(f"Top 4 Match: {q_metrics.get('top_4_match_rate', 0):.2f}%")
            print(f"Top 3 Match: {q_metrics.get('top_3_match_rate', 0):.2f}%")

            # Other bets results
            print("\nOther Bets Performance:")
            o_metrics = metrics['other_bets']
            print(f"Total Races: {o_metrics['total']}")
            print(f"Exact Order: {o_metrics.get('exact_order_rate', 0):.2f}%")
            print(f"Correct Horses: {o_metrics.get('correct_horses_rate', 0):.2f}%")

            return results
        else:
            print(f"\nError: {results['error']}")
            return None

    except Exception as e:
        print(f"Error running performance test: {str(e)}")
        raise


if __name__ == "__main__":
    import sys
    date = sys.argv[1] if len(sys.argv) > 1 else None
    main(date)