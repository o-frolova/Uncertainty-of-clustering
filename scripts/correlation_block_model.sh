#!/bin/bash

source env/bin/activate
export PYTHONPATH=/Users/olgafrolova/Documents/ЛАТАС/code
set -ex

OUTPUT_DIR=./data/results/CorrelationBlockModel/
FILE_NAME=correlation_block_model_all_results.csv
NUMBER_VERTICES=60
NUMBER_REPETITIONS=100
R_IN=0.8
R_OUT=0.1


echo "Running experiment..."
python scripts/correlation_block_model.py \
    --path-to-save ${OUTPUT_DIR} \
    --name-common-file ${FILE_NAME} \
    --number-vertices ${NUMBER_VERTICES} \
    --number-repetitions ${NUMBER_REPETITIONS} \
    --r-in ${R_IN} \
    --r-out ${R_OUT}

echo "Experiment completed. Results saved in ${OUTPUT_DIR}"
