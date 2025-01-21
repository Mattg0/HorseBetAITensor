import os
import sys
import subprocess
from env_setup import setup_environment

def change_to_root_directory(root_dir):
    try:
        os.chdir(root_dir)
        print(f"Changed working directory to: {os.getcwd()}")
    except FileNotFoundError:
        print(f"Error: The directory {root_dir} does not exist.")
        exit(0)

def execute_script(script_path, args=None):
    command = ['python', script_path]
    if args:
        command.extend(args)  # Add additional arguments if any
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing script {script_path}: {e}")

def get_bet_type_value(bet_type_name, bet_types):
    """Retrieve the value of the specified bet type."""
    bet_type = next((bt for bt in bet_types if bt['name'] == bet_type_name), None)
    return bet_type['value'] if bet_type else None

def main(model_name, mode,comp_to_predict=None, bet_type_name=None):
    # Load configuration
    config = setup_environment()

    # Change the current working directory to the root directory
    change_to_root_directory(config['rootdir'])

    # Determine which model to use based on the provided model name
    model = next((m for m in config['model'] if m['name'] == model_name), None)

    if model is None:
        print(f"Model '{model_name}' not found in configuration.")
        return

    # Get the bet type value if bet_type_name is provided
    bet_type_value = None
    if bet_type_name is not None:
        bet_type_value = get_bet_type_value(bet_type_name, config['bet_type'])
        if bet_type_value is None:
            print(f"Bet type '{bet_type_name}' not found in configuration.")
            return

    # Determine the script path based on the mode (train or predict)
    if mode == 'train':
        script_path = os.path.join(model['path'], model['train_script'])
        print(f"Executing training script: {script_path}")
        execute_script(script_path)
    elif mode == 'predict':
        script_path = os.path.join(model['path'], model['predict_script'])
        print(f"Executing prediction script: {script_path}")

        # Pass the bet type value as an argument if found
        args = []
        if bet_type_value is not None:
            args.append(comp_to_predict)
            args.append(str(bet_type_value))  # Convert to string if needed

        execute_script(script_path, args)
    else:
        print("Invalid mode specified. Please use 'train' or 'predict'.")
        return

if __name__ == "__main__":
    # Default values for training and prediction
    default_model = 'forest'  # Default model
    default_comp_to_predict = '1552621'  # Default competition ID
    default_bet_type_name = 'tierce'  # Default bet type

    # Check the number of arguments
    if len(sys.argv) == 3:  # Expecting 2 arguments: model_name and mode (for training)
        model_to_use = sys.argv[1]  # e.g., 'forest' or 'lstm'
        mode = sys.argv[2]           # e.g., 'train' or 'predict'
        comp_to_predict = None       # No competition ID needed for training
        bet_type_name = None         # No bet type needed for training
    elif len(sys.argv) == 5:  # Expecting 4 arguments: model_name, mode, comp_to_predict, bet_type_name (for prediction)
        model_to_use = sys.argv[1]       # e.g., 'forest' or 'lstm'
        mode = sys.argv[2]                # e.g., 'train' or 'predict'
        comp_to_predict = sys.argv[3]     # e.g., '1552621'
        bet_type_name = sys.argv[4]       # e.g., 'tierce'
    else:
        # Assign default values
        model_to_use = default_model
        mode = "train"  # Default to training mode
        comp_to_predict = None  # No competition ID for training
        bet_type_name = None  # No bet type for training

    # Call the main function with the determined values
    main(model_to_use, mode, comp_to_predict, bet_type_name)