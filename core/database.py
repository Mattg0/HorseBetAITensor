import sqlite3
from typing import Optional, List, Dict
from datetime import datetime
from pathlib import Path
from env_setup import setup_environment


class Database:
    def __init__(self, config_path: str = 'config.yaml', db_name: str = 'full'):
        """Initialize database connection using config.yaml.

        Args:
            config_path: Path to config file
            db_name: Database name from config (defaults to 'full')
        """
        # Load configuration
        self.config = setup_environment(config_path)

        # Get database path from config
        db_config = next((db for db in self.config['databases'] if db['name'] == db_name), None)
        if not db_config:
            raise ValueError(f"Database '{db_name}' not found in configuration")

        self.db_path = Path(self.config['rootdir']) / db_config['path']


    def get_connection(self) -> sqlite3.Connection:
        """Get an optimized database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.execute('PRAGMA journal_mode = WAL')
        conn.execute('PRAGMA cache_size = -2000')
        conn.execute('PRAGMA synchronous = NORMAL')
        return conn

    def get_path(self) -> Path:
        """Get the database file path."""
        return self.db_path