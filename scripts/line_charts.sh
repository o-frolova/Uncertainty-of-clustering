#!/bin/bash

FOLDER_PATH=./data/cbm_100_18_01_25
CORRELATION_NETWORK='Pearson'
MULTIVARIATE_DISTRIBUTION='student_distribution'
SAMPLE_SIZE=40
NUMBER_CLUSTERS=2
SAVE_PATH=./data/charts/test_line.png
MAX_R_OUT=0.8
STEP=0.05


python ./src/create_linecharts.py \
    --folder-path ${FOLDER_PATH} \
    --correlation-network ${CORRELATION_NETWORK} \
    --multivariate-distribution ${MULTIVARIATE_DISTRIBUTION} \
    --sample-size-of-observations ${SAMPLE_SIZE} \
    --number-clusters ${NUMBER_CLUSTERS} \
    --save-path ${SAVE_PATH} \
    --max-r-out ${MAX_R_OUT} \
    --step ${STEP}