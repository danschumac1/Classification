#!/bin/bash
# 544418
# ================================================================
# 2025-10-31
# Author: Dan Schumacher
# chmod +x ./bin/visual_prompting.sh
# nohup ./bin/visual_prompting.sh > ./logs/visual_prompting.log 2>&1 &
# tail -f ./logs/visual_prompting.log
# ./bin/visual_prompting.sh
# ================================================================

DATASETS=(
    # "har"
    # "ctu"
    "emg"
    # "tee"

    # CURRENTLY UNAVAILABLE
    # "rwc"   
    # "ecg"

)

MODELS=(
    "gpt-4o-mini"
)


INCLUDE_TS_BOOLS=(
    0
    # 1
)

NORMALIZATION_BOOLS=(
    0
    # 1
)

VIZ_METHODS=(
    "line"
    # "spectrogram"
)

IMG_DETAILS=(
    "auto"
    # "low"
    # "high"
)

SHOTS=(
  # 0
  # 1
  # 2
  3
  # 4
  # 5
)

# ---- Global options ----
BATCH_SIZE=5
TEMPERATURE=0.0
# ------------------------

IN_ROOT="./data/samples"

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for n_shots in "${SHOTS[@]}"; do
      for include_ts in "${INCLUDE_TS_BOOLS[@]}"; do
        for normalize in "${NORMALIZATION_BOOLS[@]}"; do
          for viz in "${VIZ_METHODS[@]}"; do
            for img_detail in "${IMG_DETAILS[@]}"; do

              echo "=============================================================="
              echo "Dataset:       $dataset"
              echo "Model:         $model"
              echo "Include TS:    $include_ts"
              echo "Normalize:     $normalize"
              echo "Viz method:    $viz"
              echo "Img detail:    $img_detail"
              echo "=============================================================="

              # ---------- Run prompting script ----------
              python ./src/visual_prompting.py \
                --input_folder "${IN_ROOT}/${dataset}/" \
                --n_shots "$n_shots" \
                --model_name "$model" \
                --batch_size "$BATCH_SIZE" \
                --temperature "$TEMPERATURE" \
                --include_ts "$include_ts" \
                --normalize "$normalize" \
                --img_detail "$img_detail" \
                --visualization_method "$viz" \
                --print_to_console 1 \
                --clear_images 1 \
                --stop_early_idx 30
                wait $!
                echo "Prompting complete."

              # if [[ $normalize -eq 1 ]]; then
              #   NORM_TAG="normalized"
              # else
              #   NORM_TAG="raw"
              # fi
              # TAG="${viz}_ID-${img_detail}_TS-${include_ts}_NORM-${NORM_TAG}"
              # OUT_JSONL="./Classification/data/sample_generations/${dataset}/visual_prompting/${TAG}.jsonl"

              # echo "Expecting JSONL at: $OUT_JSONL"

              # ---------- Run evaluation ----------
              # python ./Classification/src/eval.py \
              #   --pred_path "$OUT_JSONL"
            done
          done
        done
      done
    done
  done
done

printf "\n\nFILE DONE RUNNING 🎉🎉🎉\n\n"
