import os
import wandb
import datetime

import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Run sweep")

# Add the arguments
parser.add_argument('--env_name', type=str, required=True, help='The environment name')
args = parser.parse_args()


now = datetime.datetime.now()
current_time = now.strftime("%Y_%m_%d_%H_%M_%S")

# Load the WANDB_API_KEY environment variable
api_key = os.environ.get('WANDB_API_KEY')

# Login to wandb
wandb.login(key=api_key)


sweep_config = {
    'name': f'{args.env_name}_{current_time}',
    'method': 'random',
    'metric': {
        'name': 'test_cost',
        'goal': 'minimize'
    },
    'parameters': {
        'sink_temp': {
            'values': [0.00001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        },
        'straight_through_min': {
            'values': [True, False]
        },
        'env_temp': {
            'values': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project='ppg_sweep')


import subprocess

# Assuming args.env_name and sweep_id are defined
env_name = args.env_name
sweep_id = sweep_id

print(sweep_id)

processes = []

for i in range(2):#range(1, 7):
    command = f"python sweep_train_policy.py --env_name={env_name} --sweep_id={sweep_id}"
    process = subprocess.Popen(command, shell=True)
    processes.append(process)

# Wait for all processes to finish
for process in processes:
    process.wait()