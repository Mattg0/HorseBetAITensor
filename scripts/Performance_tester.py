import pandas as pd
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import numpy as np
from scipy import stats
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from env_setup import setup_environment, get_model_paths


@dataclass
class PredictionResult:
    """Store prediction results for a single race."""
    course_comp: int
    predictions: str
    actual_results: str
    exact_match: bool
    unordered_match: bool
    correct_numbers: int
    total_numbers: int
    confidence_scores: List[float]  # Added confidence scores
    timestamp: datetime


class PerformanceTester:
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the performance tester."""
        self.config = setup_environment(config_path)
        self.db_path = self._get_db_path()

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

    def fetch_test_courses(self, limit: int = None, offset: int = 0) -> pd.DataFrame:
        """Fetch test course data from database."""
        query = "SELECT comp, arriv FROM Courses_test"
        if limit is not None:
            query += f" LIMIT {offset},{limit}"

        with self._connect_to_db() as conn:
            return pd.read_sql_query(query, conn)

    def _parse_results(self, results_str: str) -> List[str]:
        """Parse race results string into list of positions."""
        return results_str.split('-')

    def evaluate_prediction(
            self,
            prediction: str,
            actual: str,
            confidence_scores: List[float],
            bet_type: int
    ) -> Tuple[List[bool], List[bool], List[float]]:
        """Evaluate a single prediction against actual results.

        Returns:
            Tuple containing:
            - List of exact position matches
            - List of number matches (any position)
            - List of confidence scores
        """
        predicted_list = self._parse_results(prediction)
        actual_list = self._parse_results(actual)[:bet_type]

        # Pad shorter lists if necessary
        while len(predicted_list) < bet_type:
            predicted_list.append('0')
        while len(actual_list) < bet_type:
            actual_list.append('0')

        # Check exact position matches
        exact_matches = [pred == act for pred, act in zip(predicted_list, actual_list)]

        # Check number matches in any position
        number_matches = [pred in actual_list for pred in predicted_list]

        # Ensure confidence scores are valid
        if not confidence_scores or len(confidence_scores) < bet_type:
            confidence_scores = [50.0] * bet_type  # Default confidence if not provided

        return exact_matches, number_matches, confidence_scores[:bet_type]

    def run_predictions(
            self,
            predictor_func,
            bet_type: int = 3,
            limit: Optional[int] = None,
            progress_callback=None
    ) -> List[PredictionResult]:
        """Run predictions on test courses and evaluate results."""
        courses = self.fetch_test_courses(limit=limit)
        results = []

        total = len(courses)
        for idx, (_, row) in enumerate(courses.iterrows()):
            try:
                # Get prediction and confidence scores
                prediction, confidence_scores = predictor_func(row['comp'], bet_type)

                exact, unordered, correct, total_nums, conf_scores = self.evaluate_prediction(
                    prediction, row['arriv'], confidence_scores, bet_type
                )

                results.append(PredictionResult(
                    course_comp=row['comp'],
                    predictions=prediction,
                    actual_results=row['arriv'],
                    exact_match=exact,
                    unordered_match=unordered,
                    correct_numbers=correct,
                    total_numbers=total_nums,
                    confidence_scores=conf_scores,
                    timestamp=datetime.now()
                ))

                if progress_callback:
                    progress_callback(idx + 1, total)

            except Exception as e:
                print(f"Error processing course {row['comp']}: {str(e)}")
                continue

        return results

    def analyze_confidence_correlation(self, results: List[PredictionResult]) -> Dict:
        """Analyze correlation between confidence scores and prediction accuracy."""
        if not results:
            return {
                'correlation': 0.0,
                'p_value': 1.0,
                'confidence_thresholds': {'high': 0.0, 'medium': 0.0, 'low': 0.0},
                'performance_by_confidence': {'high': 0.0, 'medium': 0.0, 'low': 0.0}
            }

        confidence_scores = []
        accuracies = []

        for result in results:
            if result.confidence_scores:  # Check if we have valid confidence scores
                # Get average confidence score for each prediction
                avg_confidence = np.mean(result.confidence_scores)
                accuracy = result.correct_numbers / result.total_numbers if result.total_numbers > 0 else 0

                confidence_scores.append(avg_confidence)
                accuracies.append(accuracy)

        if not confidence_scores or not accuracies:
            return {
                'correlation': 0.0,
                'p_value': 1.0,
                'confidence_thresholds': {'high': 0.0, 'medium': 0.0, 'low': 0.0},
                'performance_by_confidence': {'high': 0.0, 'medium': 0.0, 'low': 0.0}
            }

        # Handle case where all values are identical
        if len(set(confidence_scores)) == 1 or len(set(accuracies)) == 1:
            correlation = 0.0
            p_value = 1.0
        else:
            try:
                # Calculate correlation coefficient with error handling
                correlation, p_value = stats.pearsonr(confidence_scores, accuracies)
                # Handle NaN values
                correlation = 0.0 if np.isnan(correlation) else correlation
                p_value = 1.0 if np.isnan(p_value) else p_value
            except Exception:
                correlation = 0.0
                p_value = 1.0

        # Calculate confidence score thresholds
        confidence_thresholds = {
            'high': np.percentile(confidence_scores, 75) if confidence_scores else 0.0,
            'medium': np.percentile(confidence_scores, 50) if confidence_scores else 0.0,
            'low': np.percentile(confidence_scores, 25) if confidence_scores else 0.0
        }

        # Calculate performance by confidence level with error handling
        conf_scores = np.array(confidence_scores)
        acc_scores = np.array(accuracies)

        try:
            high_conf_mask = conf_scores >= confidence_thresholds['high']
            med_conf_mask = (conf_scores >= confidence_thresholds['medium']) & (
                        conf_scores < confidence_thresholds['high'])
            low_conf_mask = conf_scores < confidence_thresholds['low']

            performance_by_confidence = {
                'high': float(np.mean(acc_scores[high_conf_mask])) if any(high_conf_mask) else 0.0,
                'medium': float(np.mean(acc_scores[med_conf_mask])) if any(med_conf_mask) else 0.0,
                'low': float(np.mean(acc_scores[low_conf_mask])) if any(low_conf_mask) else 0.0
            }
        except Exception:
            performance_by_confidence = {'high': 0.0, 'medium': 0.0, 'low': 0.0}

        return {
            'correlation': float(correlation),
            'p_value': float(p_value),
            'confidence_thresholds': confidence_thresholds,
            'performance_by_confidence': performance_by_confidence
        }

    def calculate_metrics(self, results: List[PredictionResult]) -> Dict:
        """Calculate performance metrics from results."""
        # Return default metrics if no results
        if not results:
            return {
                'total_predictions': 0,
                'exact_matches': 0,
                'unordered_matches': 0,
                'avg_correct_numbers': 0.0,
                'success_rate': 0.0,
                'exact_match_rate': 0.0,
                'unordered_match_rate': 0.0,
                'correlation': 0.0,
                'p_value': 1.0,
                'confidence_thresholds': {'high': 0.0, 'medium': 0.0, 'low': 0.0},
                'performance_by_confidence': {'high': 0.0, 'medium': 0.0, 'low': 0.0}
            }

        total = len(results)
        base_metrics = {
            'total_predictions': total,
            'exact_matches': sum(1 for r in results if r.exact_match),
            'unordered_matches': sum(1 for r in results if r.unordered_match),
            'avg_correct_numbers': sum(r.correct_numbers for r in results) / total,
            'success_rate': sum(r.correct_numbers for r in results) / sum(r.total_numbers for r in results) * 100
        }

        base_metrics['exact_match_rate'] = (base_metrics['exact_matches'] / total) * 100
        base_metrics['unordered_match_rate'] = (base_metrics['unordered_matches'] / total) * 100

        # Add confidence analysis
        confidence_metrics = self.analyze_confidence_correlation(results)
        base_metrics.update(confidence_metrics)

        return base_metrics

    def generate_report(self, results: List[PredictionResult]) -> str:
        """Generate a formatted performance report focusing on prediction accuracy."""
        if not results:
            return "No prediction results available."

        total_races = len(results)

        # Calculate overall statistics
        exact_positions_by_rank = []  # Track exact matches for each position
        numbers_found_by_rank = []  # Track number matches for each position
        confidence_by_rank = []  # Track confidence for each position

        max_positions = len(results[0].exact_position_matches)
        for pos in range(max_positions):
            exact_matches = sum(1 for r in results if r.exact_position_matches[pos])
            number_matches = sum(1 for r in results if r.number_matches[pos])
            avg_confidence = np.mean([r.confidence_scores[pos] for r in results])

            exact_positions_by_rank.append(exact_matches)
            numbers_found_by_rank.append(number_matches)
            confidence_by_rank.append(avg_confidence)

        # Calculate average overall confidence
        avg_overall_confidence = np.mean([r.overall_confidence for r in results])

        # Generate report
        report = [
            "Horse Race Prediction Performance Report",
            "=" * 50,
            f"Total Races Analyzed: {total_races}",
            f"Average Overall Bet Confidence: {avg_overall_confidence:.1f}%",
            "\nPrediction Accuracy by Position:",
            "-" * 50
        ]

        for pos in range(max_positions):
            report.extend([
                f"\nPosition {pos + 1}:",
                f"  Exact Position Matches: {exact_positions_by_rank[pos]} ({exact_positions_by_rank[pos] / total_races * 100:.1f}%)",
                f"  Number Found Anywhere: {numbers_found_by_rank[pos]} ({numbers_found_by_rank[pos] / total_races * 100:.1f}%)",
                f"  Average Position Confidence: {confidence_by_rank[pos]:.1f}%"
            ])

        report.extend([
            "\nDetailed Results:",
            "-" * 50
        ])

        for result in results[:10]:  # Show first 10 results
            exact_matches = sum(result.exact_position_matches)
            number_matches = sum(result.number_matches)
            prediction_numbers = self._parse_results(result.predictions)
            actual_numbers = self._parse_results(result.actual_results)

            detail = f"\nCourse {result.course_comp}:"
            detail += f"\n  Overall Bet Confidence: {result.overall_confidence:.1f}%"
            detail += f"\n  Predicted: {result.predictions}"
            detail += f"\n  Actual:    {result.actual_results}"
            detail += f"\n  Exact Positions: {exact_matches}/{len(result.exact_position_matches)}"
            detail += f"\n  Numbers Found: {number_matches}/{len(result.number_matches)}"
            detail += "\n  Individual Position Confidence:"

            for pos, (pred, conf) in enumerate(zip(prediction_numbers, result.confidence_scores)):
                detail += f"\n    {pred}: {conf:.1f}%"

            report.append(detail)

        return "\n".join(report)


def main(model_name: str = 'claude', bet_type: str = 'tierce', limit: int = 10):
    """Main function to run performance testing."""
    # Initialize tester
    tester = PerformanceTester()

    # Map bet type to number
    bet_type_map = {'tierce': 3, 'quarte': 4, 'quinte': 5}
    bet_type_num = bet_type_map.get(bet_type.lower())
    if not bet_type_num:
        raise ValueError(f"Invalid bet type: {bet_type}")

    try:
        # Set up Python path for imports
        model_paths = get_model_paths(tester.config, model_name)
        model_dir = Path(model_paths['base'])
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))

        # Import the HorseRacePredictor class
        from claude_predict_race import HorseRacePredictor

        # Initialize predictor
        predictor = HorseRacePredictor()

        # Create prediction function wrapper
        def predict_wrapper(comp_id, bet_type_num):
            try:
                results = predictor.predict_race(comp_id, bet_type=bet_type, return_sequence_only=False)

                # Handle both DataFrame and tuple/string return types
                if isinstance(results, pd.DataFrame):
                    # Get the top N predictions
                    top_n = results.head(bet_type_num)

                    # Convert numeric columns to float, replacing NaN with 0
                    for col in ['odds', 'predicted_rank', 'numero']:
                        if col in top_n.columns:
                            top_n[col] = pd.to_numeric(top_n[col], errors='coerce').fillna(0)

                    # Ensure numero column exists and has valid values
                    if 'numero' not in top_n.columns:
                        sequence = "-".join([str(i + 1) for i in range(bet_type_num)])
                    else:
                        sequence = "-".join(top_n['numero'].astype(int).astype(str).tolist())

                    # Calculate confidence scores
                    try:
                        odds = top_n['odds'].values
                        odds_norm = 1 / np.where(odds > 0, odds, np.inf)  # Avoid division by zero
                        odds_norm = odds_norm / np.sum(odds_norm) if np.sum(odds_norm) > 0 else np.ones_like(
                            odds_norm) / len(odds_norm)

                        ranks = top_n['predicted_rank'].values
                        pos_norm = 1 - (ranks / len(results)) if len(results) > 0 else np.ones_like(ranks) / len(ranks)

                        # Individual confidence scores
                        confidence_scores = (0.6 * odds_norm + 0.4 * pos_norm).tolist()

                        # Calculate overall bet confidence
                        # Weight earlier positions more heavily
                        position_weights = np.array([1.0 - (i * 0.15) for i in range(bet_type_num)])
                        position_weights = position_weights / position_weights.sum()
                        overall_confidence = float(np.sum(confidence_scores * position_weights) * 100)

                    except Exception as e:
                        print(f"Error calculating confidence scores: {str(e)}")
                        confidence_scores = [1 - (i / bet_type_num) for i in range(bet_type_num)]
                        overall_confidence = 50.0  # Default moderate confidence

                elif isinstance(results, tuple):
                    # If we got a tuple, first element is the sequence
                    sequence = str(results[0])  # Ensure string type
                    confidence_scores = [1 - (i / bet_type_num) for i in range(bet_type_num)]
                    overall_confidence = 50.0  # Default moderate confidence

                else:
                    # If we got a string (sequence only)
                    sequence = str(results)  # Ensure string type
                    confidence_scores = [1 - (i / bet_type_num) for i in range(bet_type_num)]
                    overall_confidence = 50.0  # Default moderate confidence

                return sequence, confidence_scores, overall_confidence

            except Exception as e:
                print(f"Error in predict_wrapper: {str(e)}")
                # Return default prediction with low confidence
                sequence = "-".join(["0"] * bet_type_num)
                confidence_scores = [0.1] * bet_type_num
                overall_confidence = 10.0  # Low confidence for error cases
                return sequence, confidence_scores, overall_confidence

        # Define progress callback
        def show_progress(current, total):
            print(f"Progress: {current}/{total} races processed", end='\r')

        # Run predictions
        results = tester.run_predictions(
            predict_wrapper,
            bet_type_num,
            limit=limit,
            progress_callback=show_progress
        )

        # Calculate metrics and generate report
        metrics = tester.calculate_metrics(results)
        report = tester.generate_report(results, metrics)

        print("\n" + report)

        return results, metrics

    except Exception as e:
        print(f"Error running performance test: {str(e)}")
        raise


if __name__ == "__main__":
    main()