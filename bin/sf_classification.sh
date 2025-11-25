#!/bin/bash
# ================================================================
# 2025-11-20
# Author: Dan Schumacher
#
# chmod +x ./bin/sf_classification.sh
# ./bin/sf_classification.sh
#
# Run in background:
# nohup ./bin/similar_feature_classification.sh \
#       > ./logs/sf_classification_master.log 2>&1 &
#
# Tail master log:
# tail -f ./logs/sf_classification_master.log
# ================================================================

DATASETS=(
    "emg"
    # "har"
    # "ctu"
    # "tee"
)

EMBED_MODELS=(
    "text-embedding-3-small"
    # "text-embedding-3-large"
)

TOPKS=(
    5
    # 3
    # 10
)

PRINT_TO_CONSOLE=1

###############################################################################
# CONSTANTS
###############################################################################
GENERATE_DESCRIPTIONS=1
BATCH_SIZE=20
###############################################################################
for dataset in "${DATASETS[@]}"; do
  for emb_model in "${EMBED_MODELS[@]}"; do
    for top_k in "${TOPKS[@]}"; do

      echo "=============================================================="
      echo "Similarity Feature Classification"
      echo "Dataset:        $dataset"
      echo "GD?:            $GENERATE_DESCRIPTIONS"
      echo "Embedding:      $emb_model"
      echo "Top-K:          $top_k"
      echo "=============================================================="

      python ./src/similar_feature_classification.py \
        --dataset "$dataset" \
        --generate_descriptions "$GENERATE_DESCRIPTIONS" \
        --embedding_model "$emb_model" \
        --top_k "$top_k" \
        --batch_size "$BATCH_SIZE" \
        --print_to_console "$PRINT_TO_CONSOLE"

      wait $!
      echo "SF classification complete for dataset: $dataset (k=$top_k)"

    done
  done
done

printf "\n\nFILE DONE RUNNING 🎉🎉🎉\n\n"
