import os
import sys
import subprocess
from env_setup import setup_environment


def change_to_root_directory(root_dir):
    """Change the working directory to the specified root directory."""
    try:
        os.chdir(root_dir)
        print(f"Changed working directory to: {os.getcwd()}")
    except FileNotFoundError:
        print(f"Error: The directory {root_dir} does not exist.")
        sys.exit(1)


def execute_script(script_path, args=None):
    """Execute a Python script with optional arguments."""
    command = [sys.executable, script_path]
    if args:
        command.extend(args)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing script {script_path}: {e}")
        sys.exit(1)


def get_bet_type_value(bet_type_name, bet_types):
    """Retrieve the value of the specified bet type."""
    bet_type = next((bt for bt in bet_types if bt['name'] == bet_type_name), None)
    return bet_type['value'] if bet_type else None


def get_model_config(model_name, config):
    """Get model configuration based on model name."""
    return next((m for m in config['model'] if m['name'] == model_name), None)


def validate_inputs(model_name, mode, bet_type_name, config):
    """Validate input parameters against configuration."""
    # Validate model
    model = get_model_config(model_name, config)
    if not model:
        print(f"Error: Model '{model_name}' not found in configuration.")
        return False

    # Validate mode
    valid_modes = ['train', 'predict']
    if mode not in valid_modes:
        print(f"Error: Invalid mode '{mode}'. Must be one of {valid_modes}")
        return False

    # Validate bet type if provided
    if bet_type_name:
        bet_type = next((bt for bt in config['bet_type'] if bt['name'] == bet_type_name), None)
        if not bet_type:
            print(f"Error: Bet type '{bet_type_name}' not found in configuration.")
            return False

    return True


def main(model_name, mode, comp_to_predict=None, bet_type_name=None, db_type=None):
    """Main function to handle model training and prediction."""
    # Load configuration with optional database override
    config = setup_environment()

    # Validate inputs
    if not validate_inputs(model_name, mode, bet_type_name, config):
        return

    # Change to root directory
    change_to_root_directory(config['rootdir'])

    # Get model configuration
    model = get_model_config(model_name, config)

    # Get bet type value if provided
    bet_type_value = None
    if bet_type_name:
        bet_type_value = get_bet_type_value(bet_type_name, config['bet_type'])

    # Determine script path and execute
    if mode == 'train':
        script_path = os.path.join(model['base_path'], model['train_script'])
        print(f"Executing training script: {script_path}")
        execute_script(script_path)

    elif mode == 'predict':
        script_path = os.path.join(model['base_path'], model['predict_script'])
        print(f"Executing prediction script: {script_path}")

        # Prepare arguments for prediction
        args = []
        if comp_to_predict:
            args.append(str(comp_to_predict))
        if bet_type_value:
            args.append(str(bet_type_value))

        prediction = execute_script(script_path, args)
        return prediction


if __name__ == "__main__":
    # Default values
    default_model = 'claude'
    default_comp_id = '1569792'
    default_bet_type = 'quinte'

    if len(sys.argv) == 3:  # model_name and mode only
        model_to_use = sys.argv[1]
        mode = sys.argv[2]
        comp_to_predict = None
        bet_type_name = None

    elif len(sys.argv) == 5:  # full prediction command
        model_to_use = sys.argv[1]
        mode = sys.argv[2]
        comp_to_predict = sys.argv[3]
        bet_type_name = sys.argv[4]

    else:
        print("Usage:")
        print("  Training: python main.py <model_name> train")
        print("  Prediction: python main.py <model_name> predict <comp_id> <bet_type>")
        print(f"Using defaults: model={default_model}, mode=train")
        model_to_use = default_model
        mode = "train"
        comp_to_predict = None
        bet_type_name = None

    prediction = main(model_to_use, mode, comp_to_predict, bet_type_name)