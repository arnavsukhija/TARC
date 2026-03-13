import numpy as np
import pandas as pd


def load_trajectories_to_dataframes(file_dict):
    """
    Loads trajectory data from .npy files and converts them into a dictionary of pandas DataFrames.

    Args:
        file_dict (dict): A dictionary where keys are policy names and values are .npy filenames.

    Returns:
        dict: A dictionary where keys are policy names and values are pandas DataFrames.
    """
    dataframes = {}
    print("--- Loading trajectory data... ---")

    for policy_name, filename in file_dict.items():
        try:
            # Load the NumPy array from the .npy file
            trajectory_data = np.load(filename, allow_pickle=True)
            print(f"Successfully loaded '{filename}' for policy '{policy_name}'. Shape: {trajectory_data.shape}")

            # Define column names for the DataFrame
            # Assuming the columns are [steering, throttle]
            column_names = ['steering', 'throttle']

            # Ensure the number of columns matches the data
            if trajectory_data.shape[1] != len(column_names):
                print(
                    f"Warning: Data for '{policy_name}' has {trajectory_data.shape[1]} columns, but expected {len(column_names)}. Adjust column_names if needed.")
                # Fallback to generic column names if there's a mismatch
                column_names = [f'action_{i}' for i in range(trajectory_data.shape[1])]

            # Convert the NumPy array to a pandas DataFrame
            df = pd.DataFrame(trajectory_data, columns=column_names)
            dataframes[policy_name] = df

            print(f"--- DataFrame for '{policy_name}' created ---")
            print(df.head())
            print("\n")

        except FileNotFoundError:
            print(f"Error: The file '{filename}' for policy '{policy_name}' was not found. Please check the path.")
        except Exception as e:
            print(f"An error occurred while processing '{filename}': {e}")

    return dataframes


if __name__ == '__main__':
    # --- Configuration ---
    # Update this dictionary with your policy names and corresponding .npy filenames
    files_to_process = {
        'TARC_Policy': 'final Trajectories v2/origin/Tacos5/trajectory_mpljo50q.npy',
        'Baseline_Policy': 'final Trajectories v2/origin/PPO/trajectory_72pmrwwo.npy'
    }

    # Run the function and store the resulting DataFrames
    rc_car_dataframes = load_trajectories_to_dataframes(files_to_process)

    print(rc_car_dataframes['TARC_Policy'])