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
from src.correlation_block_model import CorrelationBlockModel
from src.params import PARAMS, PARAMS_NAME


def seed_all(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class CustomParser(Tap):
    path_to_save: pathlib.Path
    name_common_file: str
    number_vertices: int
    number_repetitions: int
    r_in: float
    r_out: float


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


def correlation_block_model_experiments(args):
    results_experiments = pd.DataFrame()
    combinations, combinations_name = get_combinations()

    for combination, combination_name in tqdm(zip(combinations, combinations_name), leave=False):
        print(combination_name)
        
        cbm = CorrelationBlockModel(
            num_clusters = combination['number_clusters'],
            size_cluster = int(args.number_vertices / combination['number_clusters']),
            r_in = args.r_in,
            r_out = args.r_out
        )
        cbm = cbm.create_correlation_block_model()

        ari_score_results = []
        cluster_distribution = []
        for _ in range(args.number_repetitions):
            num_clusters, result_score = one_experiment(
                cluster_method = combination['clustering_method'],
                correlation_method = combination['correlation_network'],
                multivariate_distribution = combination['multivariate_distribution'],
                number_clusters = combination['number_clusters'],
                sample_size_of_observations = combination['sample_size_of_observations'],
                true_cov_matrix = cbm['covariance_matrix'],
                true_mean_vec = cbm['mean_vector'],
                true_labels = cbm['labels']
            )
            ari_score_results.append(result_score)
            cluster_distribution.append(num_clusters)
        combination_name['ARI'] = np.mean(ari_score_results)
        combination_name['cluster_distribution'] = (cbm['true_num_clusters'], cluster_distribution)
        results_experiments = results_experiments._append(pd.Series(combination_name), ignore_index=True)
    results_experiments.to_csv(args.path_to_save / args.name_common_file)

if __name__ == "__main__":
    seed_all()
    ARGS = CustomParser(underscores_to_dashes = True).parse_args()
    correlation_block_model_experiments(ARGS)
            