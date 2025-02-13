import os
import yaml
from pathlib import Path
from typing import Dict, Any


def setup_environment(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """Load config and set up the environment with proper path handling.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dict containing the configuration
    """
    # Load configuration


    with open(config_path, 'r', encoding='utf-8', errors='ignore') as file:
        config = yaml.safe_load(file)

    # Convert root directory to absolute path
    root_dir = Path(config['rootdir']).resolve()

    # Update config with absolute paths
    config['rootdir'] = str(root_dir)

    # Create required directories
    for dir_name in ['data', 'models', 'logs', 'cache']:
        dir_path = root_dir / config['paths'][dir_name]
        dir_path.mkdir(parents=True, exist_ok=True)

    # Update database paths
    for db in config['databases']:
        if 'path' in db:
            db['path'] = str(root_dir / db['path'])

    # Update model paths
    for model in config['model']:
        model['path'] = str(root_dir / model['path'])
        model['logs'] = str(root_dir / model['logs'])

        # Create model directories
        Path(model['path']).mkdir(parents=True, exist_ok=True)
        Path(model['logs']).mkdir(parents=True, exist_ok=True)

        # Update artifact paths
        if 'artifacts' in model:
            for key, value in model['artifacts'].items():
                model['artifacts'][key] = str(Path(model['path']) / value)

    # Change working directory to root
    os.chdir(str(root_dir))

    return config


def get_model_paths(config: Dict[str, Any], model_name: str) -> Dict[str, str]:
    """Get all paths for a specific model.

    Args:
        config: Configuration dictionary
        model_name: Name of the model to get paths for

    Returns:
        Dict containing all paths for the model
    """
    model_config = next((m for m in config['model'] if m['name'] == model_name), None)
    if not model_config:
        raise ValueError(f"Model {model_name} not found in configuration")

    return {
        'base': model_config['path'],
        'logs': model_config['logs'],
        'artifacts': model_config.get('artifacts', {})
    }


def get_database_path(config: Dict[str, Any], db_name: str) -> str:
    """Get the path for a specific database.

    Args:
        config: Configuration dictionary
        db_name: Name of the database to get path for

    Returns:
        String containing the database path
    """
    db_config = next((db for db in config['databases'] if db['name'] == db_name), None)
    if not db_config:
        raise ValueError(f"Database {db_name} not found in configuration")

    return db_config['path']