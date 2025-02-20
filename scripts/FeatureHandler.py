from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from model.Claude.features import MusiqueFeatureExtractor


class FeatureHandler:
    """Handles feature preparation and validation for race predictions."""

    def __init__(self):
        """Initialize feature handler with musique extractor."""
        self.musique_extractor = MusiqueFeatureExtractor()
        # These are last-resort defaults, only used when extraction fails
        self.default_features = {
            # Horse performance features - only used when no history
            'avg_pos_3': 99.0,
            'avg_pos_5': 99.0,
            'avg_pos_10': 99.0,
            'win_rate_3': 0.0,
            'win_rate_5': 0.0,
            'win_rate_10': 0.0,
            'place_rate_3': 0.0,
            'place_rate_5': 0.0,
            'place_rate_10': 0.0,

            # Jockey features - only used when no jockey history
            'jockey_win_rate': 0.5,
            'jockey_place_rate': 0.5,
            'jockey_avg_pos': 50.0
        }

    def extract_musique_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from musique string using existing extractor.
        Falls back to defaults only if extraction fails.
        """
        df_copy = df.copy()

        for index, row in df_copy.iterrows():
            try:
                # Use existing musique feature extractor
                musique_features = self.musique_extractor.extract_features(row['musiqueche'])
                for feature, value in musique_features.items():
                    df_copy.at[index, feature] = value
            except Exception as e:
                print(f"Failed to extract musique features for horse {row.get('idche', 'unknown')}: {e}")
                # Only set defaults if extraction failed
                df_copy.at[index, 'musique_avg_position'] = 99.0
                df_copy.at[index, 'musique_top_3_rate'] = 0.0
                df_copy.at[index, 'musique_fault_prone'] = 0.0

        return df_copy

    def calculate_historical_stats(self, df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate historical statistics for horses if historical data is available.
        Only uses defaults if no history exists.
        """
        df_with_stats = df.copy()

        for horse_id in df['idche'].unique():
            horse_history = history_df[history_df['idche'] == horse_id]

            if not horse_history.empty:
                horse_history = horse_history.sort_values('jour')

                # Calculate rolling statistics
                for window in [3, 5, 10]:
                    # Position averages
                    avg_pos = horse_history['position'].rolling(window=window, min_periods=1).mean()
                    df_with_stats.loc[df_with_stats['idche'] == horse_id, f'avg_pos_{window}'] = \
                        avg_pos.iloc[-1] if not avg_pos.empty else self.default_features[f'avg_pos_{window}']

                    # Win rates
                    win_rate = (horse_history['position'] == 1).rolling(window=window, min_periods=1).mean()
                    df_with_stats.loc[df_with_stats['idche'] == horse_id, f'win_rate_{window}'] = \
                        win_rate.iloc[-1] if not win_rate.empty else self.default_features[f'win_rate_{window}']

                    # Place rates
                    place_rate = (horse_history['position'] <= 3).rolling(window=window, min_periods=1).mean()
                    df_with_stats.loc[df_with_stats['idche'] == horse_id, f'place_rate_{window}'] = \
                        place_rate.iloc[-1] if not place_rate.empty else self.default_features[f'place_rate_{window}']
            else:
                # Only use defaults if no history exists
                print(f"No history found for horse {horse_id}, using default values")
                for stat in ['avg_pos', 'win_rate', 'place_rate']:
                    for window in [3, 5, 10]:
                        feature = f"{stat}_{window}"
                        df_with_stats.loc[df_with_stats['idche'] == horse_id, feature] = \
                            self.default_features[feature]

        return df_with_stats

    def calculate_jockey_stats(self, df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate jockey statistics from historical data.
        Only uses defaults if no jockey history exists.
        """
        df_with_stats = df.copy()

        for jockey_id in df['idJockey'].unique():
            jockey_history = history_df[history_df['idJockey'] == jockey_id]

            if not jockey_history.empty:
                # Calculate actual jockey statistics
                jockey_history = jockey_history.sort_values('jour')
                recent_rides = jockey_history.tail(20)  # Last 20 rides

                df_with_stats.loc[df_with_stats['idJockey'] == jockey_id, 'jockey_win_rate'] = \
                    (recent_rides['position'] == 1).mean()
                df_with_stats.loc[df_with_stats['idJockey'] == jockey_id, 'jockey_place_rate'] = \
                    (recent_rides['position'] <= 3).mean()
                df_with_stats.loc[df_with_stats['idJockey'] == jockey_id, 'jockey_avg_pos'] = \
                    recent_rides['position'].mean()
            else:
                # Only use defaults if no jockey history exists
                print(f"No history found for jockey {jockey_id}, using default values")
                for stat in ['jockey_win_rate', 'jockey_place_rate', 'jockey_avg_pos']:
                    df_with_stats.loc[df_with_stats['idJockey'] == jockey_id, stat] = \
                        self.default_features[stat]

        return df_with_stats

    def prepare_features(self, df: pd.DataFrame, history_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare all features for prediction, prioritizing actual data over defaults.
        """
        # Start with musique feature extraction
        prepared_df = self.extract_musique_features(df)

        # Add historical stats if available
        if history_df is not None:
            prepared_df = self.calculate_historical_stats(prepared_df, history_df)
            prepared_df = self.calculate_jockey_stats(prepared_df, history_df)
        else:
            print("No historical data provided, performance metrics will use default values")
            # Set defaults only when no history is available
            for feature, default_value in self.default_features.items():
                if feature not in prepared_df.columns:
                    prepared_df[feature] = default_value

        return prepared_df

    def verify_features(self, df: pd.DataFrame, required_features: List[str]) -> bool:
        """Verify all required features are present and valid."""
        missing_features = []
        invalid_features = []

        for feature in required_features:
            if feature not in df.columns:
                missing_features.append(feature)
            elif df[feature].isna().any():
                invalid_features.append(feature)

        if missing_features:
            print(f"Missing required features: {', '.join(missing_features)}")
        if invalid_features:
            print(f"Features containing invalid values: {', '.join(invalid_features)}")

        return not (missing_features or invalid_features)