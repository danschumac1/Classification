#!/bin/bash
# 1889415
# chmod +x ./bin/generate_summaries.sh
# ./bin/generate_summaries.sh
# nohup ./bin/generate_summaries.sh > ./logs/master_generate_summaries.log 2>&1 &
# tail -f ./logs/master_generate_summaries.log

# -------------------------------------------------------------------------------------------------
# ITERABLES
DATASETS=(
    # "har"
    "emg"
    # "ctu"
    # "tee"
    # "rwc"
)

SUMMARY_TYPES=(
    "text"         # passes time series directly
    # "vis"          # passes visualization of time series
    # "letsc_like"   # passes formatted time series like LETS-C
)

# -------------------------------------------------------------------------------------------------
# ITERABLES
vis_method="line" # (line or spectrogram)
model_name="gpt-4o-mini"
temperature=0
normalize=1
batch_size=20
read_first=1

for dataset in "${DATASETS[@]}"; do
    for summary_type in "${SUMMARY_TYPES[@]}"; do
        python ./src/generate_summaries.py \
            --dataset $dataset \
            --summary_type  $summary_type \
            --vis_method $vis_method \
            --model_name $model_name \
            --temperature $temperature \
            --normalize $normalize \
            --batch_size $batch_size \
            --read_first $read_first
    done
done

