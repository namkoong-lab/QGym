#!/bin/bash

# Array of folder names
folders=(
  "reentrant_2_hyper" "reentrant_3_hyper" "reentrant_4_hyper" "reentrant_5_hyper" "reentrant_6_hyper" "reentrant_7_hyper"
  "re-reentrant_2_hyper" "re-reentrant_3_hyper" "re-reentrant_4_hyper" "re-reentrant_5_hyper" "re-reentrant_6_hyper" "re-reentrant_7_hyper"
)

# Content templates
max_pressure_content() {
  folder=$1
  echo "env: '${folder}.yaml'
model: 'ppg_linearassignment.yaml'
script: 'fixed_arrival_rate_max_pressure.py'
experiment_name: '${folder}_max_pressure'"
}

fluid_content() {
  folder=$1
  echo "env: '${folder}.yaml'
model: 'ppg_linearassignment.yaml'
script: 'fixed_arrival_rate_fluid.py'
experiment_name: '${folder}_fluid'"
}

cmuq_content() {
  folder=$1
  echo "env: '${folder}.yaml'
model: 'ppg_linearassignment.yaml'
script: 'fixed_arrival_rate_cmuq.py'
experiment_name: '${folder}_cmuq'"
}

cmu_content() {
  folder=$1
  echo "env: '${folder}.yaml'
model: 'ppg_linearassignment.yaml'
script: 'fixed_arrival_rate_cmu.py'
experiment_name: '${folder}_cmu'"
}

# Create folders and files
for folder in "${folders[@]}"; do
  mkdir -p "$folder"
  echo "$(max_pressure_content $folder)" > "$folder/max_pressure_${folder}.yaml"
  echo "$(fluid_content $folder)" > "$folder/fluid_${folder}.yaml"
  echo "$(cmuq_content $folder)" > "$folder/cmuq_${folder}.yaml"
  echo "$(cmu_content $folder)" > "$folder/cmu_${folder}.yaml"
done

echo "Folder structure and files created successfully."

