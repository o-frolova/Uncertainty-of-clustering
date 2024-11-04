#!/bin/bash

source env/bin/activate
export PYTHONPATH=/Users/olgafrolova/Documents/ЛАТАС/code
set -ex

START_DATE=2016-01-01
END_DATE=2018-12-31
PATH_TO_DATA=./data/DataStocks/SP100
PATH_TO_SAVE=./data/results/SP100_60_stocks_41124/
NAME_COMMON_FILE=SP100_60_stocks_all_results_test.csv
NUMBER_STOCKS=60
NUMBER_REPETITIONS=100

echo "Running experiment..."
python scripts/stock_markets_experiments.py \
  --start-date ${START_DATE} \
  --end-date ${END_DATE} \
  --path-to-data ${PATH_TO_DATA} \
  --path-to-save ${PATH_TO_SAVE} \
  --name-common-file ${NAME_COMMON_FILE} \
  --number-stocks ${NUMBER_STOCKS} \
  --number-repetitions ${NUMBER_REPETITIONS}

echo "Experiment completed. Results saved in ${PATH_TO_SAVE}"
