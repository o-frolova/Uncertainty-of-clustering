#!/bin/bash

set -ex

DATA_PATH="./data/SP100_60_stocks_all_results_1000_rep_18_01_25.csv"
SAVE_FOLDER="./data/tables"
BASE_NAME="SP100_60_1000"

python3 ./src/create_tables.py \
    --data-path ${DATA_PATH} \
    --path-to-save-folder ${SAVE_FOLDER} \
    --base-name-file ${BASE_NAME}
