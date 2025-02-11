import pandas as pd
import numpy as np
import hashlib
from typing import List, Dict, Optional


class DataPreprocessor:
    """Handles consistent data preprocessing across the application."""

    def __init__(self):
        self.categorical_columns = ['natpis', 'typec', 'meteo', 'corde']
        self.numeric_columns = ['dist', 'temperature', 'age', 'cotedirect']

    def preprocess_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Safely convert categorical values to numeric using hashing.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with converted categorical columns
        """
        df = df.copy()

        for col in self.categorical_columns:
            if col in df.columns:
                # First ensure column is string type
                df[col] = df[col].astype(str)

                # Fill NA values
                df[col] = df[col].fillna('unknown')

                # Hash values to numeric
                df[col] = df[col].apply(
                    lambda x: int(hashlib.md5(str(x).encode()).hexdigest(), 16) % (10 ** 8)
                )

                # Convert to float32 explicitly
                df[col] = df[col].astype('float32')
            else:
                # Add missing column with default hash
                default_hash = int(hashlib.md5('unknown'.encode()).hexdigest(), 16) % (10 ** 8)
                df[col] = float(default_hash)

        return df

    def preprocess_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Safely convert and clean numeric columns.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with cleaned numeric columns
        """
        df = df.copy()

        for col in self.numeric_columns:
            if col in df.columns:
                # Convert to numeric, coerce errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')

                # Fill NaN with median for the column
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)

                # Convert to float32
                df[col] = df[col].astype('float32')
            else:
                # Add missing column with 0
                df[col] = 0.0

        return df

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all preprocessing steps to DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Fully preprocessed DataFrame
        """
        df = self.preprocess_categorical(df)
        df = self.preprocess_numeric(df)
        return df


# Example usage in predict2.py:
def predict2(comp_id: str, bet_type: int) -> str:
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor()

        # Load and preprocess data
        next_race_data = json.loads(fetch_next_race(comp_id))
        df_next_race = pd.DataFrame(next_race_data['participants'])

        # Add course information
        if 'course_info' in next_race_data:
            course_info = next_race_data['course_info']
            for field in ['jour', 'natpis', 'typec', 'meteo', 'dist', 'corde']:
                df_next_race[field] = course_info.get(field, None)

        # Apply preprocessing
        df_next_race = preprocessor.preprocess_dataframe(df_next_race)

        # Continue with rest of prediction logic...

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return ""