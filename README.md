# QGym: Scalable Simulation and Benchmarking of Queuing Network Controllers

## Overview
QGym is an open-source simulation framework designed to benchmark queuing policies across diverse and realistic problem instances. The framework supports a wide range of environments including parallel servers, criss-cross, tandem, and re-entrant networks. It provides a platform for comparing both model-free RL methods and classical queuing policies.

## Features
- **OpenAI Gym Interface**: Easy deployment of RL algorithms.
- **Event-driven Simulation**: Precise timekeeping and fast simulation.
- **Job-Level Tracking**: Allows modeling parallel server systems.
- **Arbitrary Arrival Patterns**: Simulates time-varying arrival patterns.
- **Server Pool**: Fast simulation for a large number of same-class servers.
- **Batch Simulation**: Efficient parallel simulation of multiple trajectories.
- **Open-sourced**: Adaptable to custom needs.

<br/>

The following section details how to use the simulator to run experiments.

## Experiment Configuration

Experiments are configured using YAML files located in the [configs/experiments](vscode-remote://ssh-remote%2Bresearchgpu04.gsb.columbia.edu/user/hc3295/queue-learning/main/run_experiments.py#19%2C29-19%2C29) directory. Each experiment has its own subdirectory containing one or more YAML files specifying the environment, model, and script to run.

An example experiment YAML file:

```yaml
env: 'reentrant_5.yaml'
model: 'ppg_linearassignment.yaml'
script: 'fixed_arrival_rate_cmuq.py'
experiment_name: 'reentrant_5_cmuq'
```

## Running Experiments

To run a batch of experiments, use the `run_experiments.py` script:

```bash
python main/run_experiments.py -exp_dir=<experiment_directory>
```

Replace `<experiment_directory>` with the name of the subdirectory in `configs/experiments` containing the desired experiment YAML files.

The script will:
1. Load all YAML files in the specified experiment directory
2. For each experiment configuration:
   - Extract the environment, model, script, and experiment name
   - Launch a subprocess to run the experiment script with the specified arguments
3. Wait for all experiment subprocesses to complete

## Experiment Scripts

The experiment scripts are located in the `configs/scripts` directory. They load the environment and model configurations, set up the simulator, and run the specified routing policy.

An example experiment script:

```python
import yaml
import argparse
from main.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('-e', type=str)
parser.add_argument('-m', type=str) 
parser.add_argument('-experiment_name', type=str)

args = parser.parse_args()

with open(f'../configs/env/{args.e}', 'r') as f:
    env_config = yaml.safe_load(f)

with open(f'../configs/model/{args.m}', 'r') as f:  
    model_config = yaml.safe_load(f)

experiment_name = args.experiment_name

# Load environment and model configurations
# ...

# Run the experiment
# ...
```

## Environment and Model Configurations

The environment and model configurations are stored in YAML files in the `configs/env` and `configs/model` directories, respectively.

These files specify parameters such as the network topology, arrival rates, service rates, and routing policy hyperparameters.

## Viewing Results

Experiment results are saved in the `results` directory, organized by experiment name. Each experiment subdirectory contains log files and performance metrics.
