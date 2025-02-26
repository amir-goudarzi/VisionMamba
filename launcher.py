import itertools
import os
import subprocess
from tqdm import tqdm
tqdm(disable=True)


# Hyperparameter ranges
models = ["vim_tiny_patch28_112_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual",
          "vim_tiny_patch28_112_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual_with_cls_token",
          "vim_tiny_patch8_112_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual",
          "vim_tiny_patch8_112_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual_with_cls_token",
          "vim_small_patch28_112_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual",
          "vim_base_patch28_112_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual"
          ]

learning_rates = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]

param_combinations = list(itertools.product(
    models, learning_rates
))

# Fixed parameters
batch_size = 16
optimizer = "Adam"
scheduler = "" # False
num_epochs = 500
num_workers = 1


dir = "./"
dataset_path = "../VisualSudokuData"
neptune_project = "GRAINS/visual-sudoku"
neptune_api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly\
9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkMDQ2YThmNS1jNWU0LTQxZDItYTQxNy1lMGYzNTM4MmY5YTgifQ=="
split = 10

# File to track completed experiments
completed_experiments_file = "completed_experiments.txt"

# Load completed experiments
if os.path.exists(completed_experiments_file):
    with open(completed_experiments_file, "r") as file:
        completed_experiments = set(file.read().splitlines())
else:
    completed_experiments = set()

timeout_seconds = 720
# Loop over each combination
for params in param_combinations:

    model, learning_rate = params


    # Create a unique identifier for the experiment
    experiment_id = f"{model}_{learning_rate}"

    # Skip completed experiments
    if experiment_id in completed_experiments:
        continue


    # Construct command
    command = f"""
    python ./main.py \
      --model "{model}" \
      --batch_size "{batch_size}" \
      --optimizer "{optimizer}" \
      --learning_rate "{learning_rate}" \
      --scheduler "{scheduler}" \
      --num_epochs "{num_epochs}" \
      --num_workers "{num_workers}" \
      --dir "{dir}" \
      --dataset_path "{dataset_path}" \
      --neptune_project "{neptune_project}" \
      --neptune_api_token "{neptune_api_token}" \
      --split {split}
    """

    
    try:
        # Execute the command with a timeout
        process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_seconds)

        # Print output
        print(process.stdout)

        # Print errors (if any)
        if process.stderr:
            print("ERROR:", process.stderr)

        print("Process completed with exit code:", process.returncode)
        with open(completed_experiments_file, "a") as file:
            file.write(experiment_id + "\n")
            completed_experiments.add(experiment_id) 
    except subprocess.TimeoutExpired:
        print(f"Experiment timed out after {timeout_seconds} seconds.")
    