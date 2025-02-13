import pandas as pd
import numpy as np
import re
from typing import List, Dict, Union
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass


class MusiqueFeatureExtractor:
    @staticmethod
    def extract_positions_from_musique(musique_str: str) -> List[int]:
        """Extract only position numbers from musique string."""
        if pd.isna(musique_str) or musique_str == '':
            return []

        # Convert to string and split
        musique_parts = str(musique_str).split()
        positions = []

        for part in musique_parts:
            # Skip year markers
            if part.startswith('(') and part.endswith(')'):
                continue

            # Extract number
            match = re.search(r'(\d+)', part)
            if match:
                try:
                    pos = int(match.group(1))
                    positions.append(min(pos, 99))
                except ValueError:
                    continue

        return positions

    @staticmethod
    def extract_disciplines_from_musique(musique_str: str) -> List[str]:
        """Extract discipline letters from musique string."""
        if pd.isna(musique_str) or musique_str == '':
            return []

        disciplines = []
        for part in str(musique_str).split():
            match = re.search(r'([a-z])', part)
            if match:
                disciplines.append(match.group(1))

        return disciplines

    @staticmethod
    def extract_faults_from_musique(musique_str: str) -> List[str]:
        """Extract fault codes from musique string."""
        if pd.isna(musique_str) or musique_str == '':
            return []

        FAULT_CODES = {'A', 'D', 'T', 'RET'}
        faults = []
        for part in str(musique_str).split():
            matches = re.findall(r'([A-Z]+)', part)
            faults.extend(f for f in matches if f in FAULT_CODES)

        return faults

    def extract_features(self, musique_str: str) -> Dict[str, float]:
        """Extract musique features and return as dictionary."""
        positions = self.extract_positions_from_musique(musique_str)
        disciplines = self.extract_disciplines_from_musique(musique_str)
        faults = self.extract_faults_from_musique(musique_str)

        features = {
            'musique_total_races': float(len(positions)) if positions else 0.0,
            'musique_avg_position': float(np.mean(positions)) if positions else 99.0,
            'musique_top_3_rate': float(sum(1 for p in positions if p <= 3) / len(positions)) if positions else 0.0,
            'musique_recent_position': float(positions[0]) if positions else 99.0,
            'musique_fault_prone': float(len(faults) / len(positions) if positions else 0) > 0.2
        }

        # Add discipline features
        if disciplines:
            main_discipline = max(set(disciplines), key=disciplines.count)
            features[f'discipline_{main_discipline}'] = 1.0
        else:
            features['discipline_unknown'] = 1.0

        # Add fault features
        if faults:
            features[f'fault_{faults[0]}'] = 1.0
        else:
            features['fault_none'] = 1.0

        return features


class FeatureEngineering:
    def __init__(self):
        """Initialize feature engineering components."""
        self.position_history = {}  # Store horse performance history
        self.jockey_stats = {}  # Store jockey statistics
        self.musique_extractor = MusiqueFeatureExtractor()  # Initialize musique extractor
        import multiprocessing
        self.n_jobs = max(1, multiprocessing.cpu_count() - 1)  # Use all cores except one

    def _parallel_horse_stats(self, group_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate horse statistics in parallel."""
        horse_id = group_data['idche'].iloc[0]
        sorted_data = group_data.sort_values('jour')

        # Preallocate arrays for better performance
        n_rows = len(sorted_data)
        stats = pd.DataFrame(index=sorted_data.index)

        for window in [3, 5, 10]:
            # Use numpy's built-in functions for faster computation
            pos_array = sorted_data['position'].values
            avg_pos = pd.Series(
                np.convolve(pos_array, np.ones(window) / window, mode='full')[:n_rows],
                index=sorted_data.index
            )
            win_rate = pd.Series(
                np.convolve(pos_array == 1, np.ones(window) / window, mode='full')[:n_rows],
                index=sorted_data.index
            )
            place_rate = pd.Series(
                np.convolve(pos_array <= 3, np.ones(window) / window, mode='full')[:n_rows],
                index=sorted_data.index
            )

            stats[f'avg_pos_{window}'] = avg_pos
            stats[f'win_rate_{window}'] = win_rate
            stats[f'place_rate_{window}'] = place_rate

        return stats

    def calculate_horse_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Calculate horse-specific features with parallel processing."""
        if not is_training:
            return df

        features = df.copy()
        features['jour'] = pd.to_datetime(features['jour'])

        # Group data once to avoid multiple groupby operations
        grouped = features.groupby('idche')

        # Process groups in parallel
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_horse = {
                executor.submit(self._parallel_horse_stats, group): horse_id
                for horse_id, group in grouped
            }

            # Collect results
            all_stats = []
            for future in future_to_horse:
                stats = future.result()
                all_stats.append(stats)

        # Combine results
        combined_stats = pd.concat(all_stats)
        features = features.join(combined_stats, how='left')

        return features

    def calculate_jockey_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Calculate jockey-specific features with vectorized operations."""
        if not is_training:
            return df.assign(
                jockey_win_rate=0.0,
                jockey_place_rate=0.0,
                jockey_avg_pos=50.0
            )

        features = df.copy()
        features['jour'] = pd.to_datetime(features['jour'])

        # Sort by jockey and date first
        features = features.sort_values(['idJockey', 'jour'])

        # Create a grouped object with sorted index
        grouped = features.groupby('idJockey')

        # Calculate rolling statistics
        rolling_stats = pd.DataFrame(index=features.index)

        # Average position
        rolling_stats['jockey_avg_pos'] = grouped['position'].transform(
            lambda x: x.rolling(window=20, min_periods=1).mean()
        )

        # Win rate
        rolling_stats['jockey_win_rate'] = grouped['position'].transform(
            lambda x: (x == 1).rolling(window=20, min_periods=1).mean()
        )

        # Place rate
        rolling_stats['jockey_place_rate'] = grouped['position'].transform(
            lambda x: (x <= 3).rolling(window=20, min_periods=1).mean()
        )

        # Add statistics to features
        for col in rolling_stats.columns:
            features[col] = rolling_stats[col]

        return features

        # Merge back efficiently
        features = pd.merge(
            features,
            jockey_stats,
            on=['idJockey', 'jour'],
            how='left'
        )

        return features

    def calculate_race_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate race-specific features with vectorized operations."""
        features = df.copy()

        # Calculate all race features at once
        features['normalized_odds'] = (
            features.groupby('comp')['cotedirect']
            .transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))
        )

        features['odds_rank'] = (
            features.groupby('comp')['cotedirect']
            .transform('rank')
        )

        # Use numpy for age categorization
        age_bins = [-np.inf, 3, 5, 7, np.inf]
        age_labels = ['young', 'emerging', 'mature', 'veteran']
        features['age_category'] = pd.cut(
            features['age'],
            bins=age_bins,
            labels=age_labels
        ).astype(str)

        return features

    def extract_all_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Extract and combine all features."""
       # print("Starting feature extraction...")
        features = df.copy()

        try:
            # Extract musique features for each row
       #     print("Extracting musique features...")
            musique_features_list = []
            for musique in features['musiqueche']:
                musique_features = self.musique_extractor.extract_features(musique)
                musique_features_list.append(musique_features)

            # Convert musique features to DataFrame
            musique_df = pd.DataFrame(musique_features_list, index=features.index)
            features = pd.concat([features, musique_df], axis=1)

            # Calculate additional features
       #     print("Calculating horse features...")
            features = self.calculate_horse_features(features, is_training)

       #     print("Calculating jockey features...")
            features = self.calculate_jockey_features(features, is_training)

       #     print("Calculating race features...")
            features = self.calculate_race_features(features)

        except Exception as e:
       #     print(f"Error during feature extraction: {str(e)}")
            raise

       # print("Feature extraction completed successfully")
        return features

    def get_feature_columns(self) -> List[str]:
        """Return list of feature columns used for modeling."""
        base_features = [
            'age', 'cotedirect', 'dist', 'temperature',
            'natpis', 'typec', 'meteo', 'corde'
        ]

        musique_features = [
            'musique_total_races', 'musique_avg_position', 'musique_top_3_rate',
            'musique_recent_position', 'musique_fault_prone'
        ]

        horse_features = [
            'avg_pos_3', 'avg_pos_5', 'avg_pos_10',
            'win_rate_3', 'win_rate_5', 'win_rate_10',
            'place_rate_3', 'place_rate_5', 'place_rate_10'
        ]

        jockey_features = [
            'jockey_win_rate', 'jockey_place_rate', 'jockey_avg_pos'
        ]

        race_features = [
            'normalized_odds', 'odds_rank'
        ]

        return base_features + musique_features + horse_features + jockey_features + race_features