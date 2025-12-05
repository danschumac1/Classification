#!/bin/bash
# 2081668
# ================================================================
# 2025-12-04
# Logistic Regression Baseline Launcher
# Author: Dan Schumacher
#
# chmod +x ./bin/logistic_regression.sh
# ./bin/logistic_regression.sh
# nohup ./bin/logistic_regression.sh > ./logs/logistic_regression_master.log 2>&1 &
# tail -f ./logs/logistic_regression_master.log
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
    # SINGLE FEATURES
    # "ts_direct"
    # "letsc_direct"
    # "vis_direct"
    # "text_summary"
    # "letsc_summary"
    # "vis_summary"
    # MULTI-FEATURES
    "ts_direct-text_summary"
    "vis_direct-vis_summary"
    "letsc_direct-letsc_summary"
    "ts_direct-vis_direct"
    "text_summary-letsc_summary-vis_summary"
    "ts_direct-text_summary-letsc_summary-vis_summary"



)

# Logistic regression regularization strengths
C_VALUES=( # must be floats (include decimal, or will fail)
    0.1 
    1.0
    10.0
)

NORMALIZE=1      # 1 = normalize, 0 = no normalization
METHOD="logistic_regression"  # for eval logging

###############################################################################
# LOOP
###############################################################################

for dataset in "${DATASETS[@]}"; do
    for e_type in "${EMBED_TYPES[@]}"; do
        for C in "${C_VALUES[@]}"; do
            echo "=============================================================="
            echo "Logistic Regression Pipeline"
            echo "Dataset:        $dataset"
            echo "Embedding Type: $e_type"
            echo "C (reg):        $C"
            echo "Normalize:      $NORMALIZE"
            echo "=============================================================="

            # ------------------------------
            # Run logistic regression script
            # ------------------------------
            python ./src/logistic_regression.py \
                --dataset "$dataset" \
                --embedding_type "$e_type" \
                --C "$C" \
                --normalize "$NORMALIZE"

            status=$?
            if [[ $status -ne 0 ]]; then
                echo "❌ Error: logistic_regression.py failed for dataset=$dataset, embedding_type=$e_type, C=$C"
                exit 1
            fi

            echo "logistic_regression complete → dataset=$dataset  type=$e_type  C=$C"
            echo ""

            # ------------------------------
            # Run evaluation on logistic_regression outputs

            C_SAFE="${C//./p}"   # replace "." with "p" for safe filenames

            OUT_JSONL="./data/sample_generations/${dataset}/logistic_regression/${e_type}/${C_SAFE}.jsonl"
            MODE="${e_type}-C${C}"

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
                echo "❌ Error: eval.py failed for dataset=$dataset, embedding_type=$e_type, C=$C"
                exit 1
            fi

            echo "Eval complete → dataset=$dataset  type=$e_type  C=$C"
            echo ""

            wait $!
        done
    done
done

printf "\n\nFILE DONE RUNNING 🎉🎉🎉\n\n"
