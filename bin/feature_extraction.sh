#!/bin/bash
# ================================================================
# 2025-10-31
# Author: Dan Schumacher
# chmod +x ./bin/feature_extraction.sh
# ./bin/feature_extraction.sh
# nohup ./bin/feature_extraction.sh > ./logs/feature_extraction.log 2>&1 &
# tail -f ./logs/feature_extraction.log
# ================================================================

DATASETS=(
    "emg"
    # "har"
    # "ctu"
    # "tee"
)

MODELS=(
    "gpt-4o-mini"
)


NORMALIZATION_BOOLS=(
    0
    # 1
)

VIZ_METHODS=(
    "line"
    # "spectrogram"
)

# ---- Global options ----
BATCH_SIZE=20
TEMPERATURE=0.0
PRINT_TO_CONSOLE=1
# ------------------------

IN_ROOT="./data/samples"

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for normalize in "${NORMALIZATION_BOOLS[@]}"; do
      for viz in "${VIZ_METHODS[@]}"; do

        echo "=============================================================="
        echo "Dataset:       $dataset"
        echo "Model:         $model"
        echo "Normalize:     $normalize"
        echo "Viz method:    $viz"
        echo "=============================================================="

        python ./src/feature_extraction.py \
          --input_folder "${IN_ROOT}/${dataset}/" \
          --visualization_method "$viz" \
          --model_name "$model" \
          --temperature "$TEMPERATURE" \
          --batch_size "$BATCH_SIZE" \
          --normalize "$normalize" \
          --print_to_console "$PRINT_TO_CONSOLE"

        wait $!
        echo "Feature extraction complete for dataset: $dataset"

      done
    done
  done
done

printf "\n\nFILE DONE RUNNING 🎉🎉🎉\n\n"
