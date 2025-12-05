#!/bin/bash
# 1896370
# ================================================================
# 2025-12-04
# KNN Baseline Launcher
# Author: Dan Schumacher
#
# chmod +x ./bin/knn.sh
# ./bin/knn.sh
# nohup ./bin/knn.sh > ./logs/knn_master.log 2>&1 &
# tail -f ./logs/knn_master.log
# ================================================================

###############################################################################
# CONFIGURATION
###############################################################################
DATASETS=(
    "har"
    "emg"
    "ctu"
    "tee"
    # "rwc" # LATER
)

EMBED_TYPES=(
    "ts_direct"
    "letsc_direct"
    "vis_direct"
    "text_summary"
    "letsc_summary"
    "vis_summary"
    # "ts_special" # LATER
)

K_VALUES=(
    1
    3
    5
    7
    10
)

NORMALIZE=1     # 1 = normalize, 0 = no normalization

METHOD="knn"    # for eval logging

###############################################################################
# LOOP
###############################################################################

for dataset in "${DATASETS[@]}"; do
  for e_type in "${EMBED_TYPES[@]}"; do
    for k in "${K_VALUES[@]}"; do

      echo "=============================================================="
      echo "KNN Pipeline"
      echo "Dataset:        $dataset"
      echo "Embedding Type: $e_type"
      echo "k:              $k"
      echo "Normalize:      $NORMALIZE"
      echo "=============================================================="

      # ------------------------------
      # Run KNN script
      # ------------------------------
      python ./src/knn.py \
        --dataset "$dataset" \
        --embedding_type "$e_type" \
        --k "$k" \
        --normalize "$NORMALIZE"

      status=$?
      if [[ $status -ne 0 ]]; then
          echo "❌ Error: knn.py failed for dataset=$dataset, embedding_type=$e_type, k=$k"
          exit 1
      fi

      echo "KNN complete → dataset=$dataset  type=$e_type  k=$k"
      echo ""

      # ------------------------------
      # Run evaluation on KNN outputs
      #
      # Assumes knn.py writes to:
      #   outdir = ./data/sample_generations/{dataset}/knn/{embedding_type}/k{k}
      #   outpath = {outdir}/{embedding_type}-k{k}.jsonl
      # ------------------------------
      OUT_JSONL="./data/sample_generations/${dataset}/knn/${e_type}/k${k}.jsonl"
      MODE="${e_type}-k${k}"

      echo "Evaluating predictions at: $OUT_JSONL"
      echo "  dataset = ${dataset}"
      echo "  method  = ${METHOD}"
      echo "  mode    = ${MODE}"

      python ./src/eval.py \
        --dataset "$dataset" \
        --method "$METHOD" \
        --mode "$MODE" \
        --pred_path "$OUT_JSONL"

      eval_status=$?
      if [[ $eval_status -ne 0 ]]; then
          echo "❌ Error: eval.py failed for dataset=$dataset, embedding_type=$e_type, k=$k"
          exit 1
      fi

      echo "Eval complete → dataset=$dataset  type=$e_type  k=$k"
      echo ""

      wait $!

    done
  done
done

printf "\n\nFILE DONE RUNNING 🎉🎉🎉\n\n"
