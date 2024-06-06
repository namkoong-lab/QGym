import os
import subprocess

# Directory containing the environment configuration files
env_dir = "configs/env"

# Get a list of all environment configuration files
env_files = [f for f in os.listdir(env_dir) if f.endswith(".yaml") and not f.startswith("debug")]

processes = []

# Run start_sweeps.py for each environment configuration file
for env_file in env_files:
    env_name = os.path.splitext(env_file)[0]  # Remove the .yaml extension
    command = f"python start_sweeps.py --env_name={env_name}.yaml"
    process = subprocess.Popen(command, shell=True)
    processes.append(process)

# Wait for all processes to finish
for process in processes:
    process.wait()