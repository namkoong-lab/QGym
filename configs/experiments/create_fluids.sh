#!/bin/bash

# List of filenames
filenames=(
    "reentrant_10.yaml" "reentrant_4.yaml" "reentrant_8.yaml" "re-reentrant_4_hyper.yaml" "re-reentrant_7.yaml"
    "reentrant_2_hyper.yaml" "reentrant_5_hyper.yaml" "reentrant_9.yaml" "re-reentrant_4.yaml" "re-reentrant_8.yaml"
    "reentrant_2_varying.yaml" "reentrant_5.yaml" "re-reentrant_10.yaml" "re-reentrant_5_hyper.yaml" "re-reentrant_9.yaml"
    "reentrant_2.yaml" "reentrant_6_hyper.yaml" "re-reentrant_2_hyper.yaml" "re-reentrant_5.yaml"
    "reentrant_3_hyper.yaml" "reentrant_6.yaml" "re-reentrant_2.yaml" "re-reentrant_6_hyper.yaml"
    "reentrant_3.yaml" "reentrant_7_hyper.yaml" "re-reentrant_3_hyper.yaml" "re-reentrant_6.yaml"
    "reentrant_4_hyper.yaml" "reentrant_7.yaml" "re-reentrant_3.yaml" "re-reentrant_7_hyper.yaml"
)

# Template content
content_template() {
    local name=$1
    echo "env: '${name}'
model: 'ppg_linearassignment.yaml'
script: 'fixed_arrival_rate_fluid.py'
experiment_name: 'new-${name}_fluid'"
}

# Loop through the filenames and create the files with the content
for filename in "${filenames[@]}"; do
    name="${filename%.yaml}"
    content=$(content_template "$name")
    echo "$content" > "fluid_${filename}"
done

echo "Files created successfully."
