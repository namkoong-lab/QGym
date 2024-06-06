folders=(
  "reentrant_2_hyper" "reentrant_3_hyper" "reentrant_4_hyper" "reentrant_5_hyper" "reentrant_6_hyper" "reentrant_7_hyper"
  "re-reentrant_2_hyper" "re-reentrant_3_hyper" "re-reentrant_4_hyper" "re-reentrant_5_hyper" "re-reentrant_6_hyper" "re-reentrant_7_hyper"
)

for folder in "${folders[@]}"; do
  rm -rf "${folder}.out"
done


for folder in "${folders[@]}"; do
  nohup python run_experiments.py -exp_dir="$folder" > "${folder}.out" &
done
