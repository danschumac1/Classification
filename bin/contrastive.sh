#!/bin/bash
# ================================================================
# 2025-11-21
# 2371143
# Author: Dan Schumacher
#
# chmod +x ./bin/contrastive.sh
# ./bin/contrastive.sh
#
# Run in background:
# nohup ./bin/contrastive.sh > ./logs/contrastive_master.log 2>&1 &
#
# Tail master log:
# tail -f ./logs/contrastive_master.log
# ================================================================

DATASETS=(
    # "har"
    # "emg"
    # "ctu"
    "tee"
)

N_IMGS=(
    10
    # 5
    # 3
    # 1
)

PRINT_TO_CONSOLE=1

MODEL="gpt-4o-mini"
TEMPERATURE=0.0
IMG_DETAIL="auto"

###############################################################################
# LOOP
###############################################################################
for dataset in "${DATASETS[@]}"; do
  for nimg in "${N_IMGS[@]}"; do

    echo "=============================================================="
    echo "Contrastive Feature Summaries"
    echo "Dataset:          $dataset"
    echo "Images/Class:     $nimg"
    echo "Model:            $MODEL"
    echo "Temp:             $TEMPERATURE"
    echo "Detail:           $IMG_DETAIL"
    echo "=============================================================="

    python ./src/contrastive.py \
      --dataset "$dataset" \
      --n_imgs "$nimg" \
      --print_to_console "$PRINT_TO_CONSOLE" \
      --model_name "$MODEL" \
      --temperature "$TEMPERATURE" \
      --img_detail "$IMG_DETAIL" \

    wait $!
    echo "Contrastive summaries complete → $dataset (n_imgs=$nimg)"

  done
done

printf "\n\nFILE DONE RUNNING 🎉🎉🎉\n\n"
