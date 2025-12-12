#!/bin/bash
# 2770368
# ================================================================
# 2025-10-31
# Author: Dan Schumacher
# chmod +x ./bin/llamaPrompting.sh
# nohup ./bin/llamaPrompting.sh > ./logs/llamaPrompting.log 2>&1 &
# tail -f ./logs/llamaPrompting.log
# ./bin/llamaPrompting.sh
# ================================================================

DATASETS=(
  "har"
  "ctu"
  "emg"
  "tee"
)

MODELS=(
  "meta-llama/Llama-3.2-11B-Vision-Instruct"
  "meta-llama/Llama-3.2-11B-Vision"
)

SHOTS=(
  0
  3
  5
)

BATCH_SIZE=10

# Single GPU index; change to "0,1" etc. if you want multiple visible devices
GPUS=2

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for n_shots in "${SHOTS[@]}"; do

      echo "=============================================================="
      echo "Dataset:       $dataset"
      echo "Model:         $model"
      echo "n_shots:       $n_shots"
      echo "Batch size:    $BATCH_SIZE"
      echo "GPU(s):        $GPUS"
      echo "=============================================================="

      CUDA_VISIBLE_DEVICES=$GPUS python ./src/llamaPrompting.py \
        --dataset "$dataset" \
        --model "$model" \
        --batch_size "$BATCH_SIZE" \
        --n_shots "$n_shots"

      echo "Prompting complete for dataset=${dataset}, model=${model}, n_shots=${n_shots}."
      echo

    done
  done
done

printf "\n\nFILE DONE RUNNING 🎉🎉🎉\n\n"
