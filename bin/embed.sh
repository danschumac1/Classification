#!/bin/bash
# 1896370
# ================================================================
# 2025-12-03
# Embedding Pipeline Launcher
# Author: Dan Schumacher
#
# chmod +x ./bin/embed.sh
# ./bin/embed.sh
# nohup ./bin/embed.sh > ./logs/embed_master.log 2>&1 &
# tail -f ./logs/embed_master.log
# ================================================================

###############################################################################
# CONFIGURATION
###############################################################################
DATASETS=(
    # "har"
    "emg"
    # "ctu"
    # "tee"
    # "rwc" # LATER
)

EMBED_TYPES=(
    # DIRECT EMBEDDING
    # "ts_direct"
    "letsc_direct"
    "vis_direct"
    # "ts_special" # TODO

    # SUMMARY EMBEDDINGS
    "text_summary"
    "letsc_summary"
    "vis_summary"
)

# Default embed model: # not used for vision
EMBED_MODEL="text-embedding-3-large"    
CLIP_MODEL="clip-ViT-L/14"    
VIS_METHOD="line"               
CUDA_DEVICES="2,3"
NORMALIZE=1
BATCH_SIZE=1
###############################################################################
# LOOP
###############################################################################

for dataset in "${DATASETS[@]}"; do

  for e_type in "${EMBED_TYPES[@]}"; do
    # Select correct embedding model for the chosen embedding type
    MODEL_FLAG="$EMBED_MODEL"
    if [[ "$e_type" == "vis_direct" ]]; then
      MODEL_FLAG="$CLIP_MODEL"
    fi

    echo "=============================================================="
    echo "Embedding Pipeline"
    echo "Dataset:        $dataset"
    echo "Embed Type:     $e_type"
    echo "Model:          $MODEL_FLAG"
    echo "Batch Size:     $BATCH_SIZE"
    echo "=============================================================="


    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python ./src/embed.py \
      --dataset "$dataset" \
      --embedding_type "$e_type" \
      --embed_model "$MODEL_FLAG" \
      --vis_method "$VIS_METHOD" \
      --normalize $NORMALIZE \
      --batch_size $BATCH_SIZE

    status=$?
    if [[ $status -ne 0 ]]; then
        echo "❌ Error: embed.py failed for dataset=$dataset, embedding_type=$e_type"
        exit 1
    fi

    echo "Embedding complete → dataset=$dataset  type=$e_type"
    echo ""

    wait $!
  done
done

printf "\n\nFILE DONE RUNNING 🎉🎉🎉\n\n"
