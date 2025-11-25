#!/bin/bash
# ================================================================
# 2025-11-20
# Author: Dan Schumacher
#
# Unified pipeline:
#   1. Train description generation (optional)
#   2. Test description generation (optional)
#   3. Embeddings + similarity-based classification
#
# chmod +x ./bin/sf_pipeline.sh
# ./bin/sf_pipeline.sh
#
# Run in background:
#   nohup ./bin/sf_pipeline.sh > ./logs/sf_pipeline_master.log 2>&1 &
#
# Tail logs:
#   tail -f ./logs/sf_pipeline_master.log
# ================================================================

DATASETS=(
    "emg"
    # "har"
    # "ctu"
    # "tee"
)

VISION_MODELS=(
  "gpt-4o-mini"
  # "gpt-4o"
)

EMBED_MODELS=(
  "text-embedding-3-small"
  # "text-embedding-3-large"
)

NORMALIZATION_BOOLS=(
  0
  # 1
)

VIZ_METHODS=(
  "line"
  # "spectrogram"
)

TOPKS=(
  # 1
  3
  # 5
  10
)

# ----------------------------------------------------------------
# Global options
# ----------------------------------------------------------------
BATCH_SIZE=20
TEMPERATURE=0.0
PRINT_TO_CONSOLE=1

# Regeneration flags:
REGEN_TRAIN=0         # 0 = reuse if exists, 1 = regenerate train descriptions
REGEN_TEST=0          # 0 = reuse if exists, 1 = regenerate test descriptions
# ----------------------------------------------------------------

IN_ROOT="./data/samples"

for dataset in "${DATASETS[@]}"; do
  for vmodel in "${VISION_MODELS[@]}"; do
    for emb_model in "${EMBED_MODELS[@]}"; do
      for normalize in "${NORMALIZATION_BOOLS[@]}"; do
        for viz in "${VIZ_METHODS[@]}"; do
          for top_k in "${TOPKS[@]}"; do

            echo "=============================================================="
            echo "Unified SF Pipeline"
            echo "Dataset:          $dataset"
            echo "Vision model:     $vmodel"
            echo "Embedding model:  $emb_model"
            echo "Normalize:        $normalize"
            echo "Viz:              $viz"
            echo "Top-K:            $top_k"
            echo "Regen Train:      $REGEN_TRAIN"
            echo "Regen Test:       $REGEN_TEST"
            echo "=============================================================="

            python ./src/sf_pipeline.py \
              --dataset "$dataset" \
              --input_folder "${IN_ROOT}/${dataset}/" \
              --visualization_method "$viz" \
              --model_name "$vmodel" \
              --embedding_model "$emb_model" \
              --top_k "$top_k" \
              --batch_size "$BATCH_SIZE" \
              --temperature "$TEMPERATURE" \
              --normalize "$normalize" \
              --regen_train_desc "$REGEN_TRAIN" \
              --regen_test_desc "$REGEN_TEST" \
              --print_to_console "$PRINT_TO_CONSOLE"

            wait $!
            echo "Unified SF pipeline complete for dataset: $dataset (k=$top_k)"
            echo

          done
        done
      done
    done
  done
done

printf "\n\nFILE DONE RUNNING 🎉🎉🎉\n\n"
