#!/bin/bash
# ================================================================
# 2025-11-25
# chmod +x ./bin/_get_raw_data.sh
# ./bin/_get_raw_data.sh
# ================================================================

# CTU
mkdir -p data/raw_data/ctu
wget -O data/raw_data/ctu/CTU_TRAIN.ts https://raw.githubusercontent.com/AdityaLab/TimerBed/main/Datasets/CTU/CTU_TRAIN.ts
wget -O data/raw_data/ctu/CTU_TEST.ts  https://raw.githubusercontent.com/AdityaLab/TimerBed/main/Datasets/CTU/CTU_TEST.ts

# EMG
mkdir -p data/raw_data/emg
wget -O data/raw_data/emg/EMG_TRAIN.ts https://raw.githubusercontent.com/AdityaLab/TimerBed/main/Datasets/EMG/EMG_TRAIN.ts
wget -O data/raw_data/emg/EMG_TEST.ts  https://raw.githubusercontent.com/AdityaLab/TimerBed/main/Datasets/EMG/EMG_TEST.ts

# HAR
mkdir -p data/raw_data/har
wget -O data/raw_data/har/HAR_TRAIN.ts https://raw.githubusercontent.com/AdityaLab/TimerBed/main/Datasets/HAR/HAR_TRAIN.ts
wget -O data/raw_data/har/HAR_TEST.ts  https://raw.githubusercontent.com/AdityaLab/TimerBed/main/Datasets/HAR/HAR_TEST.ts

# TEE
mkdir -p data/raw_data/tee
wget -O data/raw_data/tee/TEE_TRAIN.ts https://raw.githubusercontent.com/AdityaLab/TimerBed/main/Datasets/TEE/TEE_TRAIN.ts
wget -O data/raw_data/tee/TEE_TEST.ts  https://raw.githubusercontent.com/AdityaLab/TimerBed/main/Datasets/TEE/TEE_TEST.ts
