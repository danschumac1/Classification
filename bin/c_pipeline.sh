#!/bin/bash
# 2379813
# chmod +x ./bin/c_pipeline.sh
# ./bin/c_pipeline.sh
# nohup ./bin/c_pipeline.sh > ./logs/c_pipeline_master.log 2>&1 &
# tail -f ./logs/c_pipeline_master.log
DATASETS=(
    "ctu"
    "emg"
    # "har"
    # "tee"
)
for dataset in "${DATASETS[@]}"; do
    python ./src/c_pipeline.py \
        --dataset $dataset \
        --contrastive_input_path ./data/features/$dataset/10-imgs_5-rounds_contrastive.json
done
