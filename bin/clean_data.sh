#!/bin/bash
# chmod +x ./bin/clean_data.sh
# ./bin/clean_data.sh

DATASETS=(
    "ctu"
    "emg"
    "har"
    "tee"
)

for dataset in "${DATASETS[@]}"; do
    python ./src/clean_data.py \
        --dataset "$dataset"
done