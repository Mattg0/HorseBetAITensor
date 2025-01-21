import os
import yaml


def setup_environment(config_path='config.yaml'):
    """Load config and change working directory to rootdir."""
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Change the current working directory to the root directory
    root_dir = config['rootdir']
    os.chdir(root_dir)

    return config