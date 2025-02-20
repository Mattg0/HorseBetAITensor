import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def setup_environment(config_path: str = 'config.yaml', override_db: str = None) -> Dict[str, Any]:
    """Load config and set up the environment with proper path handling.

    Args:
        config_path: Path to the configuration file
        override_db: Optional override for active_db setting in config

    Returns:
        Dict containing the configuration
    """
    # Load configuration
    with open(config_path, 'r', encoding='utf-8', errors='ignore') as file:
        config = yaml.safe_load(file)

    # Use override_db if provided, otherwise use config's active_db
    active_db = override_db or config.get('active_db', 'full')
    config['active_db'] = active_db

    # Convert root directory to absolute path
    root_dir = Path(config['rootdir']).resolve()
    config['rootdir'] = str(root_dir)

    # Create required directories
    for dir_name in ['data', 'models', 'logs', 'cache']:
        dir_path = root_dir / config['paths'][dir_name]
        dir_path.mkdir(parents=True, exist_ok=True)

    # Update database paths
    for db in config['databases']:
        if 'path' in db:
            db['path'] = str(root_dir / db['path'])

    # Set active database configuration
    config['active_db_config'] = next(
        (db for db in config['databases'] if db['name'] == active_db),
        config['databases'][0]  # Default to first database if type not found
    )

    # Update model paths
    for model in config['model']:
        # Create base path
        base_path = root_dir / model['base_path']
        model['base_path'] = str(base_path)

        # Create model version paths
        for version in model['model_paths'].keys():
            version_path = base_path / model['model_paths'][version]
            model['model_paths'][version] = str(version_path)
            version_path.mkdir(parents=True, exist_ok=True)

        # Update logs path
        model['logs'] = str(root_dir / model['logs'])
        Path(model['logs']).mkdir(parents=True, exist_ok=True)

    # Update cache paths
    for cache_type in config['cache']:
        for version in config['cache'][cache_type]:
            cache_path = str(root_dir / config['cache'][cache_type][version])
            config['cache'][cache_type][version] = cache_path
            # Create cache directory
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)

    # Change working directory to root
    os.chdir(str(root_dir))

    return config


def get_model_paths(config: Dict[str, Any], model_name: str, override_db: str = None) -> Dict[str, str]:
    """Get all paths for a specific model.

    Args:
        config: Configuration dictionary
        model_name: Name of the model to get paths for
        override_db: Optional override for active database

    Returns:
        Dict containing all paths for the model
    """
    model_config = next((m for m in config['model'] if m['name'] == model_name), None)
    if not model_config:
        raise ValueError(f"Model {model_name} not found in configuration")

    # Use override_db if provided, otherwise use config's active_db
    db_type = override_db or config['active_db']

    # Get the specific model path for the database type
    model_path = model_config['model_paths'].get(db_type)
    if not model_path:
        raise ValueError(f"Database type {db_type} not configured for model {model_name}")

    return {
        'base': model_config['base_path'],
        'model_path': model_path,
        'logs': model_config['logs'],
        'artifacts': model_config.get('artifacts', {}),
        'db_type': db_type  # Include the active database type in the returned paths
    }


def get_database_path(config: Dict[str, Any], override_db: str = None) -> str:
    """Get the path for the active database.

    Args:
        config: Configuration dictionary
        override_db: Optional override for active database

    Returns:
        String containing the database path
    """
    db_type = override_db or config['active_db']
    db_config = next((db for db in config['databases'] if db['name'] == db_type), None)
    if not db_config:
        raise ValueError(f"Database {db_type} not found in configuration")

    return db_config['path']


def get_cache_path(config: Dict[str, Any], cache_type: str, override_db: str = None) -> str:
    """Get the cache path for a specific type.

    Args:
        config: Configuration dictionary
        cache_type: Type of cache ('historical_data' or 'training_data')
        override_db: Optional override for active database

    Returns:
        String containing the cache path
    """
    db_type = override_db or config['active_db']
    try:
        return config['cache'][cache_type][db_type]
    except KeyError:
        raise ValueError(f"Cache path not found for type '{cache_type}' and database '{db_type}'")