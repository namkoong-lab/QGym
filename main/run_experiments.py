import argparse
import os
import yaml

import os
import subprocess


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-exp_dir', type=str)


args = parser.parse_args()


exp_configs = []

for file in os.listdir(f'../configs/experiments/{args.exp_dir}'):
    if file.endswith('.yaml'):
        with open(f'../configs/experiments/{args.exp_dir}/{file}', 'r') as f:
            exp_configs.append(yaml.safe_load(f))

processes = []
for exp_config in exp_configs:
    script_file = exp_config['script']
    model_file = exp_config['model']
    env_file = exp_config['env']
    experiment_name = exp_config['experiment_name']

    command = f"python ../configs/scripts/{script_file} -m={model_file} -e={env_file} -experiment_name={experiment_name}"
    process = subprocess.Popen(command, shell=True)
    processes.append(process)

for process in processes:
    process.wait()

