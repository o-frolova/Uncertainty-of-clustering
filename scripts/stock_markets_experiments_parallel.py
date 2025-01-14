import itertools
import os
import pathlib
import random

import numpy as np
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
from tap import Tap
from tqdm import tqdm

from src.artificial_cluster_structure import ArtificialСlusterStructure
from src.params import PARAMS, PARAMS_NAME
from src.Stocks import Stocks
from src.StocksReader import ReaderStocksData

from joblib import Parallel, delayed

def seed_all(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class CustomParser(Tap):
    start_date: str = "2016-01-01"
    end_date: str = "2018-12-31"
    path_to_data: pathlib.Path = "./data/DataStocks/SP100_1619"
    path_to_save: pathlib.Path = "./data/results/SP100_60_stocks_10_rep/"
    name_common_file: str = "SP100_60_stocks_all_results_5_rep_test_1_1.csv"
    number_stocks: int = 20
    number_repetitions: int = 5

def load_data(args):
    ReaderData = ReaderStocksData(args.path_to_data)
    DATA_OF_STOCKS, TICKERS = ReaderData.load_data(args.start_date, args.end_date)

    selected_indices = random.sample(range(99), args.number_stocks)
    DATA_OF_STOCKS = [DATA_OF_STOCKS[i] for i in selected_indices]
    TICKERS = [TICKERS[i] for i in selected_indices]

    return DATA_OF_STOCKS, TICKERS

def get_covariance_matrix(Stocks: Stocks, tickers: np.array) -> pd.DataFrame:
    """
    Calculates the covariance matrix for the given list of stock objects.
    
    Args:
    - stocks (list): List of stock objects, each having a 'returns' attribute.
    - tickers (np.array): Array of stock tickers corresponding to the stocks.
    
    Returns:
    - pd.DataFrame: Covariance matrix.
    """
    covariance_matrix = []
    for stock_1 in Stocks:
        covv = []
        for stock_2 in Stocks:
            covv.append(np.cov(stock_1.returns, stock_2.returns)[0, 1])
        covariance_matrix.append(covv)

    return pd.DataFrame(covariance_matrix, columns = tickers, index = tickers)

def get_mean_vector(Stocks: Stocks) -> np.array:
    """
    Calculates the mean returns for the given list of stock objects.
    
    Args:
    - stocks (list): List of stock objects, each having a 'returns' attribute.
    
    Returns:
    - np.array: Array of mean returns.
    """
    mean_vector = []
    for stock in Stocks:
        mean_vector.append(stock.returns.mean())
    return np.array(mean_vector)

def get_combinations() -> tuple[list[dict], list[dict]]:
    keys, values = zip(*PARAMS.items())
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

    keys_name, values_name = zip(*PARAMS_NAME.items())
    combinations_name = [dict(zip(keys_name, combination)) for combination in itertools.product(*values_name)]
    print(f'Total number of combinations: {len(combinations)}')
    print(f'Total number of name combinations: {len(combinations_name)}')
    return combinations, combinations_name

def one_experiment(
      cluster_method,
      correlation_method,
      multivariate_distribution,
      number_clusters,
      sample_size_of_observations,
      true_cov_matrix,
      true_mean_vec,
      true_labels
) -> float:
   artificial_cluster_structure = ArtificialСlusterStructure()
   num_clusters, gen_labels_ = artificial_cluster_structure.clustering(
      multivariate_distribution = multivariate_distribution,
      mean_vector = true_mean_vec,
      cov_matrix = true_cov_matrix,
      sample_size_of_observations = sample_size_of_observations,
      correlation_method = correlation_method,
      clustering_method = cluster_method,
      number_clusters = number_clusters
   )

   return num_clusters, adjusted_rand_score(true_labels, gen_labels_)

def experiment_worker(combination, combination_name, DATA_OF_STOCKS, true_cov_matrix, true_mean_vec, args):
    print(combination_name)

    correlation_matrix = []
    for stock_1 in DATA_OF_STOCKS:
        row = []
        for stock_2 in DATA_OF_STOCKS:
            row.append(combination['correlation_network'](data_1=stock_1.returns, data_2=stock_2.returns))
        correlation_matrix.append(row)
    true_num_clusters, true_labels = combination['clustering_method'](np.array(correlation_matrix), combination['number_clusters'])

    ari_score_results = []
    cluster_distribution = []
    for _ in range(args.number_repetitions):
        num_clusters, result_score = one_experiment(
            cluster_method=combination['clustering_method'],
            correlation_method=combination['correlation_network'],
            multivariate_distribution=combination['multivariate_distribution'],
            number_clusters=combination['number_clusters'],
            sample_size_of_observations=combination['sample_size_of_observations'],
            true_cov_matrix=true_cov_matrix,
            true_mean_vec=true_mean_vec,
            true_labels=true_labels
        )
        ari_score_results.append(result_score)
        cluster_distribution.append(num_clusters)

    combination_name['ARI'] = np.mean(ari_score_results)
    combination_name['cluster_distribution'] = (true_num_clusters, cluster_distribution)
    return pd.Series(combination_name)

def stock_markets_experiments(args):
    DATA_OF_STOCKS, TICKERS = load_data(args)
    true_cov_matrix = get_covariance_matrix(DATA_OF_STOCKS, TICKERS)
    true_mean_vec = get_mean_vector(DATA_OF_STOCKS)

    combinations, combinations_name = get_combinations()

    # Parallel execution of experiments
    results_experiments = Parallel(n_jobs=2, backend='loky')(
        delayed(experiment_worker)(
            combination, combination_name, DATA_OF_STOCKS, true_cov_matrix, true_mean_vec, args
        )
        for combination, combination_name in tqdm(zip(combinations, combinations_name), leave=False)
    )

    # Combine results into a DataFrame
    results_experiments_df = pd.DataFrame(results_experiments)
    results_experiments_df.to_csv(args.path_to_save / args.name_common_file)

if __name__ == "__main__":
    seed_all()
    ARGS = CustomParser(underscores_to_dashes = True).parse_args()
    stock_markets_experiments(ARGS)
