#!/bin/bash

# Iterate from 2 to 7
for i in {2..7}
do
    env_name="reentrant_$i" # Construct the environment name
    # Use nohup to run the python script in the background for each environment
    nohup python -u train_policy_batchtest.py -e=$env_name.yaml -m=ppg.yaml > "${env_name}.out" 2>&1 &
done

# Wait for all background jobs to finish
wait

