#!/bin/bash

source env/bin/activate
export PYTHONPATH=/Users/olgafrolova/Documents/ЛАТАС/code
set -ex

PATH_TO_SAVE=./data/results/CorrelationBlockModel/
FILE_NAME=correlation_block_model_all_results.csv
NUMBER_VERTICES=60
NUMBER_REPETITIONS=100
R_IN=0.8
R_OUT=0.1

echo "Running experiment..."
python scripts/correlation_block_model.py \
    --path-to-save ${PATH_TO_SAVE} \
    --name-common-file ${FILE_NAME} \
    --number-vertices ${NUMBER_VERTICES} \
    --number-repetitions ${NUMBER_REPETITIONS} \
    --r-in ${R_IN} \
    --r-out ${R_OUT}

echo "Experiment completed. Results saved in ${PATH_TO_SAVE}"
