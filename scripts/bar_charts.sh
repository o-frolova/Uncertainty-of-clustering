#!/bin/bash

set -ex

FOLDER_PATH=./data/diff_student
CORRELATION_NETWORK='Pearson'
MULTIVARIATE_DISTRIBUTION='student_distribution'
SAMPLE_SIZE=40
NUMBER_CLUSTERS=2
SAVE_PATH=./data/charts/test.png
MAX_R_OUT=0.8
STEP=0.5


python ./src/create_barcharts.py \
    --folder-path ${FOLDER_PATH} \
    --correlation-network ${CORRELATION_NETWORK} \
    --multivariate-distribution ${MULTIVARIATE_DISTRIBUTION} \
    --sample-size-of-observations ${SAMPLE_SIZE} \
    --number-clusters ${NUMBER_CLUSTERS} \
    --save-path ${SAVE_PATH} \
    --max-r-out ${MAX_R_OUT} \
    --step ${STEP}