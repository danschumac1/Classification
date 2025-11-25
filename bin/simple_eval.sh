#!/usr/bin/env bash
# chmod +x ./bin/simple_eval.sh
# ./bin/simple_eval.sh

PATHS=(
    # "./data/sample_generations/har/visual_prompting/line-auto-no_ts-raw-3_shot.jsonl"
    # "./data/sample_generations/ctu/visual_prompting/line-auto-no_ts-raw-3_shot.jsonl"
    # "./data/sample_generations/tee/visual_prompting/line-auto-no_ts-raw-3_shot.jsonl"
    # "./data/sample_generations/emg/visual_prompting/line-auto-no_ts-raw-3_shot.jsonl"

    # "data/sample_generations/har/sf_classification/top-01.jsonl"
    # "data/sample_generations/ctu/sf_classification/top-01.jsonl"
    # "data/sample_generations/tee/sf_classification/top-01.jsonl"
    # "data/sample_generations/emg/sf_classification/top-01.jsonl"

    # "data/sample_generations/har/sf_classification/top-03.jsonl"
    # "data/sample_generations/ctu/sf_classification/top-03.jsonl"
    # "data/sample_generations/tee/sf_classification/top-03.jsonl"
    # "data/sample_generations/emg/sf_classification/top-03.jsonl"

    # "data/sample_generations/har/sf_classification/top-05.jsonl"
    # "data/sample_generations/ctu/sf_classification/top-05.jsonl"
    # "data/sample_generations/tee/sf_classification/top-05.jsonl"
    # "data/sample_generations/emg/sf_classification/top-05.jsonl"

    # "data/sample_generations/har/sf_classification/top-10.jsonl"
    # "data/sample_generations/ctu/sf_classification/top-10.jsonl"
    # "data/sample_generations/tee/sf_classification/top-10.jsonl"
    # "data/sample_generations/emg/sf_classification/top-10.jsonl"

    #  "./data/sample_generations/har/contrastive_classification/cf.jsonl"
    # "./data/sample_generations/ctu/contrastive_classification/cf.jsonl"
    # "./data/sample_generations/tee/contrastive_classification/cf.jsonl"
    "./data/sample_generations/emg/contrastive_classification/cf.jsonl"


)

for path in "${PATHS[@]}"; do
  python ./src/eval.py --pred_path "$path"
  printf "\n"
done
