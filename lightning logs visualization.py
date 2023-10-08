import pandas as pd
import matplotlib.pyplot as plt

def plot_combined_stages(file_paths):
    """
    Generate a combined plot showing the evolution of the NCC/training and NCC/validation metrics 
    over the adjusted steps for multiple stages.

    Args:
    - file_paths (list of str): List of file paths for the different stages.
    """
    plt.figure(figsize=(16, 9))
    
    # Initial max step value
    max_step_value = 0

    colors_training = ['blue', 'cyan', 'lightblue', 'royalblue', 'midnightblue']
    colors_validation = ['red', 'orange', 'yellow', 'coral', 'darkred']

    for idx, file_path in enumerate(file_paths):
        # Load the data
        df_stage = pd.read_csv(file_path)
        
        # Adjust the step values
        df_stage['adjusted_step'] = df_stage['step'] + max_step_value
        
        # Update the max step value for the next iteration
        max_step_value = df_stage['adjusted_step'].max()
        
        # Filter out rows with non-null values for each metric
        training_data_stage = df_stage.dropna(subset=['NCC/training'])
        validation_data_stage = df_stage.dropna(subset=['NCC/validation'])

        # Plot training data
        plt.plot(training_data_stage['adjusted_step'], training_data_stage['NCC/training'], 
                 label=f'NCC/training Stage {idx + 1}', 
                 color=colors_training[idx], marker='o', markersize=5)

        # Plot validation data
        plt.plot(validation_data_stage['adjusted_step'], validation_data_stage['NCC/validation'], 
                 label=f'NCC/validation Stage {idx + 1}', 
                 color=colors_validation[idx], marker='.', markersize=8, linestyle='--', linewidth=0.75)

    # Adding labels, title, and legend
    plt.xlabel('Iteration')
    plt.ylabel('NCC Value')
    plt.title('Evolution of NCC Metrics over Adjusted Steps for Multiple Stages')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Display the combined plot
    plt.tight_layout()
    plt.show()

plot_combined_stages(["D:/bunya_output/lightning_logs/version_25/metrics.csv", "D:/bunya_output/lightning_logs/version_26/metrics.csv"])
