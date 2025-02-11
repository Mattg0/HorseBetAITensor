from pathlib import Path
import pandas as pd
from typing import Optional, Union, Dict
import json
import hashlib
from datetime import datetime


class CacheManager:
    def __init__(self, cache_dir: Union[str, Path], max_age_hours: int = 24):
        """Initialize the cache manager.

        Args:
            cache_dir: Directory to store cache files
            max_age_hours: Maximum age of cache in hours before refresh
        """
        self.cache_dir = Path(cache_dir)
        self.max_age_hours = max_age_hours
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata file if it doesn't exist
        self.metadata_path = self.cache_dir / 'cache_metadata.json'
        if not self.metadata_path.exists():
            self._save_metadata({})

    def _get_cache_path(self, name: str) -> Path:
        """Get the full path for a cached file."""
        return self.cache_dir / f"{name}.parquet"

    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute a hash of the DataFrame content."""
        return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()

    def _save_metadata(self, metadata: Dict) -> None:
        """Save metadata to JSON file."""
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    def _load_metadata(self) -> Dict:
        """Load metadata from JSON file."""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {}

    def _update_metadata(self, name: str, df: pd.DataFrame) -> None:
        """Update metadata for a cache entry."""
        metadata = self._load_metadata()
        metadata[name] = {
            'timestamp': datetime.now().isoformat(),
            'rows': len(df),
            'columns': list(df.columns),
            'data_hash': self._compute_data_hash(df)
        }
        self._save_metadata(metadata)

    def save(self, df: pd.DataFrame, name: str) -> None:
        """Save DataFrame to cache with metadata.

        Args:
            df: DataFrame to cache
            name: Name of the cache entry
        """
        try:
            # Save the data
            cache_path = self._get_cache_path(name)
            df.to_parquet(str(cache_path))

            # Update metadata
            self._update_metadata(name, df)

            print(f"Cached data saved to: {cache_path}")

        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    def load(self, name: str, validate_func=None) -> Optional[pd.DataFrame]:
        """Load DataFrame from cache if valid.

        Args:
            name: Name of the cache entry
            validate_func: Optional function to validate loaded data

        Returns:
            DataFrame if valid cache exists, None otherwise
        """
        cache_path = self._get_cache_path(name)
        if not cache_path.exists():
            return None

        try:
            # Check metadata
            metadata = self._load_metadata()
            if name not in metadata:
                return None

            cache_time = datetime.fromisoformat(metadata[name]['timestamp'])
            age = datetime.now() - cache_time

            if age.total_seconds() > self.max_age_hours * 3600:
                print(f"Cache is older than {self.max_age_hours} hours, will reload data")
                return None

            # Load data
            df = pd.read_parquet(str(cache_path))

            # Verify data hash
            current_hash = self._compute_data_hash(df)
            if current_hash != metadata[name]['data_hash']:
                print("Cache data hash mismatch, will reload data")
                return None

            # Run custom validation if provided
            if validate_func and not validate_func(df):
                print("Cache validation failed, will reload data")
                return None

            print(f"Loading cached data from: {cache_path}")
            return df

        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            return None

    def clear(self, name: Optional[str] = None) -> None:
        """Clear cache entries.

        Args:
            name: Specific cache entry to clear, or None to clear all
        """
        try:
            metadata = self._load_metadata()

            if name:
                # Clear specific entry
                if name in metadata:
                    cache_path = self._get_cache_path(name)
                    if cache_path.exists():
                        cache_path.unlink()
                    del metadata[name]
                    print(f"Cleared cache entry: {name}")
            else:
                # Clear all entries
                for cache_name in metadata.keys():
                    cache_path = self._get_cache_path(cache_name)
                    if cache_path.exists():
                        cache_path.unlink()
                metadata = {}
                print("Cleared all cache entries")

            self._save_metadata(metadata)

        except Exception as e:
            print(f"Warning: Failed to clear cache: {e}")

    def get_info(self) -> Dict:
        """Get information about cached data."""
        metadata = self._load_metadata()
        info = {}

        for name, entry in metadata.items():
            cache_time = datetime.fromisoformat(entry['timestamp'])
            age = datetime.now() - cache_time

            info[name] = {
                'age_hours': age.total_seconds() / 3600,
                'rows': entry['rows'],
                'columns': entry['columns'],
                'size_mb': self._get_cache_path(name).stat().st_size / (1024 * 1024)
            }

        return info